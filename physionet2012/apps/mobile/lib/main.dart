// lib/main.dart
import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';

typedef Preset = String; // "quick" | "normal" | "detailed"

const String sampleNote = '''
55F, fever 39.1C, cough, RR 28, SpO2 90%, WBC 16.2, CRP 120, wheeze/coarse breath sounds RLL. CXR pending.

LAST_12H_WINDOW:
HR: last=46.000, min=38.000, max=62.000, mean=48.500, slope_hr=-1.100
MAP: last=64.000, min=52.000, max=74.000, mean=61.000, slope_hr=-0.500
SBP: last=92.000, min=78.000, max=108.000, mean=90.500, slope_hr=-0.700
DBP: last=50.000, min=40.000, max=64.000, mean=49.200, slope_hr=-0.400
RR: last=10.000, min=8.000, max=18.000, mean=12.200, slope_hr=-0.250
SpO2: last=98.000, min=95.000, max=100.000, mean=98.600, slope_hr=0.050
TempC: last=36.100, min=35.700, max=36.600, mean=36.200, slope_hr=-0.020
Lactate: last=1.300, min=1.100, max=2.000, mean=1.450, slope_hr=-0.080
Creatinine: last=1.100, min=0.900, max=1.200, mean=1.050, slope_hr=0.020
WBC: last=9.800, min=7.900, max=10.400, mean=9.200, slope_hr=0.040
pH: last=7.210, min=7.200, max=7.330, mean=7.270, slope_hr=-0.030
PaO2: last=84.000, min=74.000, max=98.000, mean=86.000, slope_hr=-0.600
PaCO2: last=72.000, min=58.000, max=72.000, mean=65.500, slope_hr=1.200
Na: last=138.000, min=136.000, max=140.000, mean=138.200, slope_hr=0.020
K: last=4.200, min=3.600, max=4.700, mean=4.150, slope_hr=0.060
Glucose: last=118.000, min=92.000, max=148.000, mean=121.000, slope_hr=-0.300

CONTEXT:
72F found somnolent with shallow breathing after receiving opioids for post-op pain control 6 hours ago (hydromorphone dose unclear).
Pupils reportedly pinpoint. End-tidal CO2 rising; ABG shows hypercapnia with acidemia. SpO2 preserved on 2L nasal cannula.
No known drug allergies. Concern for opioid-induced hypoventilation vs COPD/OSA overlap; consider naloxone and airway support.
''';

/// Android emulator -> host machine localhost
String get apiBase {
  if (kIsWeb) return "http://127.0.0.1:8000";
  if (defaultTargetPlatform == TargetPlatform.android) return "http://10.0.2.2:8000";
  return "http://127.0.0.1:8000";
}

/// Model generation can take minutes; don't time out aggressively.
const Duration analyzeTimeout = Duration(hours: 5);

/// Health endpoint should respond quickly.
const Duration healthTimeout = Duration(seconds: 15);

/// Remove any <...> tags + collapse huge blank gaps
String cleanModelText(String s) {
  if (s.isEmpty) return s;
  final withoutTags = s.replaceAll(RegExp(r'</?[^>]+>'), '');
  final collapsed = withoutTags.replaceAll(RegExp(r'\n{3,}'), '\n\n');
  return collapsed.trim();
}

class NeonTheme {
  static const bg = Color(0xFF0B0B10);

  static const text = Colors.white;
  static const muted = Color(0xB3FFFFFF);
  static const muted2 = Color(0x8CFFFFFF);

  static const red = Color(0xFFFF2D2D);
  static const red2 = Color(0xFFFF4D4D);

  // ✅ OPAQUE fills so the background glow does NOT bleed through
  static const boxFill = Color(0xFF14141B);
  static const boxFill2 = Color(0xFF101017);

  static BoxShadow glow({double strength = 1.0}) {
    return BoxShadow(
      color: red.withOpacity(0.35 * strength),
      blurRadius: 18 * strength,
      offset: const Offset(0, 0),
    );
  }

  static BoxShadow glowStrong() {
    return BoxShadow(
      color: red.withOpacity(0.55),
      blurRadius: 28,
      offset: const Offset(0, 0),
    );
  }
}

/// Reads Firebase web config from --dart-define.
/// Required keys for web:
/// - FIREBASE_API_KEY
/// - FIREBASE_AUTH_DOMAIN
/// - FIREBASE_PROJECT_ID
/// - FIREBASE_STORAGE_BUCKET
/// - FIREBASE_MESSAGING_SENDER_ID
/// - FIREBASE_APP_ID
FirebaseOptions? firebaseOptionsFromEnv({required bool requireAll}) {
  const apiKey = String.fromEnvironment('FIREBASE_API_KEY');
  const authDomain = String.fromEnvironment('FIREBASE_AUTH_DOMAIN');
  const projectId = String.fromEnvironment('FIREBASE_PROJECT_ID');
  const storageBucket = String.fromEnvironment('FIREBASE_STORAGE_BUCKET');
  const messagingSenderId = String.fromEnvironment('FIREBASE_MESSAGING_SENDER_ID');
  const appId = String.fromEnvironment('FIREBASE_APP_ID');

  // Optional (mostly for analytics)
  const measurementId = String.fromEnvironment('FIREBASE_MEASUREMENT_ID');

  final missing = <String>[];
  if (apiKey.isEmpty) missing.add('FIREBASE_API_KEY');
  if (authDomain.isEmpty) missing.add('FIREBASE_AUTH_DOMAIN');
  if (projectId.isEmpty) missing.add('FIREBASE_PROJECT_ID');
  if (storageBucket.isEmpty) missing.add('FIREBASE_STORAGE_BUCKET');
  if (messagingSenderId.isEmpty) missing.add('FIREBASE_MESSAGING_SENDER_ID');
  if (appId.isEmpty) missing.add('FIREBASE_APP_ID');

  if (missing.isNotEmpty) {
    if (requireAll) {
      throw Exception(
        "Missing --dart-define values:\n- ${missing.join("\n- ")}\n\n"
        "Example:\n"
        "--dart-define=FIREBASE_API_KEY=... --dart-define=FIREBASE_AUTH_DOMAIN=... etc",
      );
    }
    return null;
  }

  return FirebaseOptions(
    apiKey: apiKey,
    authDomain: authDomain,
    projectId: projectId,
    storageBucket: storageBucket,
    messagingSenderId: messagingSenderId,
    appId: appId,
    measurementId: measurementId.isEmpty ? null : measurementId,
  );
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  String? initError;

  try {
    if (kIsWeb) {
      // Web ALWAYS needs explicit options.
      final opts = firebaseOptionsFromEnv(requireAll: true)!;
      await Firebase.initializeApp(options: opts);
    } else {
      // Non-web: try platform default first (google-services.json / plist / etc).
      // If that isn’t configured, fallback to --dart-define options if provided.
      try {
        await Firebase.initializeApp();
      } catch (_) {
        final opts = firebaseOptionsFromEnv(requireAll: false);
        if (opts == null) rethrow;
        await Firebase.initializeApp(options: opts);
      }
    }
  } catch (e) {
    initError = e.toString();
  }

  runApp(MedGemmaApp(firebaseInitError: initError));
}

class MedGemmaApp extends StatelessWidget {
  const MedGemmaApp({super.key, required this.firebaseInitError});
  final String? firebaseInitError;

  @override
  Widget build(BuildContext context) {
    final theme = ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: NeonTheme.bg,
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: NeonTheme.red,
        brightness: Brightness.dark,
      ),
      textTheme: const TextTheme(
        bodyMedium: TextStyle(color: NeonTheme.text),
      ),
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: "Triage Assist-SA:",
      theme: theme,
      home: AuthGate(firebaseInitError: firebaseInitError),
    );
  }
}

class AuthGate extends StatelessWidget {
  const AuthGate({super.key, required this.firebaseInitError});
  final String? firebaseInitError;

  @override
  Widget build(BuildContext context) {
    if (firebaseInitError != null && firebaseInitError!.trim().isNotEmpty) {
      return _ConfigErrorPage(message: firebaseInitError!);
    }

    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snap) {
        if (snap.connectionState == ConnectionState.waiting) {
          return const _LoadingPage();
        }
        final user = snap.data;
        if (user == null) return const _AuthPage();
        return HomePage(user: user);
      },
    );
  }
}

/* ------------------ auth pages ------------------ */

class _LoadingPage extends StatelessWidget {
  const _LoadingPage();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: const [
          _BackgroundGlow(),
          SafeArea(
            child: Center(
              child: SizedBox(
                width: 420,
                child: _NeonCard(
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text("🔐 Loading session…",
                            style: TextStyle(fontWeight: FontWeight.w900, fontSize: 14)),
                        SizedBox(height: 10),
                        _PillDim(text: "Checking Firebase session…"),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ConfigErrorPage extends StatelessWidget {
  const _ConfigErrorPage({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          const _BackgroundGlow(),
          SafeArea(
            child: Center(
              child: SizedBox(
                width: 720,
                child: _NeonCard(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text("⚙️ Firebase init error",
                            style: TextStyle(fontWeight: FontWeight.w900, fontSize: 14)),
                        const SizedBox(height: 10),
                        _OutputBox(
                          controller: ScrollController(),
                          text: message,
                          fill: NeonTheme.boxFill2,
                          textStyle: TextStyle(color: NeonTheme.muted, fontSize: 12, height: 1.35),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          "Fix your --dart-define values (web) or platform config (mobile/desktop) then restart.",
                          style: TextStyle(color: NeonTheme.muted2, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AuthPage extends StatefulWidget {
  const _AuthPage();

  @override
  State<_AuthPage> createState() => _AuthPageState();
}

class _AuthPageState extends State<_AuthPage> {
  final emailCtrl = TextEditingController();
  final passCtrl = TextEditingController();

  bool creating = false;
  bool busy = false;
  String error = "";

  @override
  void dispose() {
    emailCtrl.dispose();
    passCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() {
      busy = true;
      error = "";
    });

    final email = emailCtrl.text.trim();
    final pass = passCtrl.text;

    if (email.isEmpty || pass.isEmpty) {
      setState(() {
        busy = false;
        error = "Enter email + password.";
      });
      return;
    }

    try {
      if (creating) {
        await FirebaseAuth.instance.createUserWithEmailAndPassword(email: email, password: pass);
      } else {
        await FirebaseAuth.instance.signInWithEmailAndPassword(email: email, password: pass);
      }
    } on FirebaseAuthException catch (e) {
      setState(() {
        error = e.message ?? e.code;
      });
    } catch (e) {
      setState(() {
        error = e.toString();
      });
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          const _BackgroundGlow(),
          SafeArea(
            child: Center(
              child: SizedBox(
                width: 520,
                child: _NeonCard(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Text(
                          creating ? "🔐 Create account" : "🔐 Sign in",
                          style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 14),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          "Demo auth gate (email/password).",
                          style: TextStyle(color: NeonTheme.muted, fontSize: 12),
                        ),
                        const SizedBox(height: 12),

                        _LabeledField(
                          label: "Email",
                          child: _NeonInput(
                            controller: emailCtrl,
                            hint: "name@example.com",
                            keyboardType: TextInputType.emailAddress,
                          ),
                        ),
                        const SizedBox(height: 10),
                        _LabeledField(
                          label: "Password",
                          child: _NeonInput(
                            controller: passCtrl,
                            hint: "••••••••",
                            obscure: true,
                          ),
                        ),

                        const SizedBox(height: 12),
                        _PrimaryButton(
                          label: busy ? "Please wait…" : (creating ? "Create account" : "Sign in"),
                          loading: busy,
                          onTap: busy ? null : _submit,
                        ),

                        AnimatedSwitcher(
                          duration: const Duration(milliseconds: 180),
                          child: error.isEmpty
                              ? const SizedBox(height: 8)
                              : Padding(
                                  padding: const EdgeInsets.only(top: 10),
                                  child: _Toast(text: error),
                                ),
                        ),

                        const SizedBox(height: 6),
                        Wrap(
                          spacing: 6,
                          crossAxisAlignment: WrapCrossAlignment.center,
                          children: [
                            Text(
                              creating ? "Already have an account?" : "Need an account?",
                              style: TextStyle(color: NeonTheme.muted, fontSize: 12),
                            ),
                            TextButton(
                              onPressed: busy
                                  ? null
                                  : () => setState(() {
                                        creating = !creating;
                                        error = "";
                                      }),
                              child: Text(
                                creating ? "Sign in" : "Create one",
                                style: TextStyle(color: NeonTheme.red2, fontWeight: FontWeight.w900),
                              ),
                            ),
                          ],
                        ),
                        Text(
                          "Users appear in Firebase Console → Authentication → Users.",
                          style: TextStyle(color: NeonTheme.muted2, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _LabeledField extends StatelessWidget {
  const _LabeledField({required this.label, required this.child});
  final String label;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
        const SizedBox(height: 6),
        child,
      ],
    );
  }
}

class _NeonInput extends StatefulWidget {
  const _NeonInput({
    required this.controller,
    required this.hint,
    this.keyboardType,
    this.obscure = false,
  });

  final TextEditingController controller;
  final String hint;
  final TextInputType? keyboardType;
  final bool obscure;

  @override
  State<_NeonInput> createState() => _NeonInputState();
}

class _NeonInputState extends State<_NeonInput> {
  final FocusNode focus = FocusNode();

  @override
  void dispose() {
    focus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final focused = focus.hasFocus;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 150),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: NeonTheme.red.withOpacity(focused ? 0.70 : 0.26)),
        color: NeonTheme.boxFill,
        boxShadow: focused ? [NeonTheme.glow()] : [],
      ),
      clipBehavior: Clip.antiAlias,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: TextField(
        controller: widget.controller,
        focusNode: focus,
        keyboardType: widget.keyboardType,
        obscureText: widget.obscure,
        style: const TextStyle(color: NeonTheme.text, height: 1.35),
        decoration: InputDecoration(
          border: InputBorder.none,
          hintText: widget.hint,
          hintStyle: const TextStyle(color: Color(0x80FFFFFF)),
        ),
      ),
    );
  }
}

/* ------------------ main app ------------------ */

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.user});
  final User user;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage>
    with WidgetsBindingObserver, SingleTickerProviderStateMixin {
  Preset preset = "quick";
  final TextEditingController noteCtrl = TextEditingController(text: sampleNote);

  String reply = "";
  Map<String, dynamic>? meta;
  Map<String, dynamic>? health;
  String error = "";
  bool loading = false;

  final ScrollController replyScroll = ScrollController();
  final ScrollController metaScroll = ScrollController();

  Timer? healthTimer;
  late final AnimationController sweepCtrl;

  // ✅ Keep a single client alive (more reliable for long calls)
  late http.Client _client;

  // Used to cancel in-flight requests
  bool _cancelRequested = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _client = http.Client();

    sweepCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 5500))
      ..repeat();

    _refreshHealth();
    _startHealthTimer();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    healthTimer?.cancel();

    noteCtrl.dispose();
    replyScroll.dispose();
    metaScroll.dispose();
    sweepCtrl.dispose();

    _client.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _refreshHealth();
      _startHealthTimer();
    } else {
      _stopHealthTimer();
    }
  }

  void _startHealthTimer() {
    healthTimer?.cancel();
    healthTimer = Timer.periodic(const Duration(seconds: 12), (_) => _refreshHealth());
  }

  void _stopHealthTimer() {
    healthTimer?.cancel();
    healthTimer = null;
  }

  String get presetHint {
    if (preset == "quick") return "⚡ Fast triage bullets (brief).";
    if (preset == "normal") return "📋 SOAP + tasks + red flags + patient summary.";
    return "🧠 Full analysis with workup + management considerations.";
  }

  bool get apiReady => (health?["ok"] == true);

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), duration: const Duration(milliseconds: 1100)),
    );
  }

  // ---------- Networking (robust for long responses) ----------

  Future<Map<String, dynamic>> _httpGetJson(
    String url, {
    Duration timeout = healthTimeout,
  }) async {
    final uri = Uri.parse(url);
    final res = await _client.get(uri).timeout(timeout);
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw Exception("HTTP ${res.statusCode}: ${res.body}");
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  /// Robust POST that handles long-running responses better than `client.post()`.
  Future<Map<String, dynamic>> _httpPostJsonLong(
    String url,
    Map<String, dynamic> body, {
    Duration timeout = analyzeTimeout,
  }) async {
    final sw = Stopwatch()..start();
    final uri = Uri.parse(url);

    final req = http.Request("POST", uri);
    req.headers.addAll({
      "Content-Type": "application/json",
      "Accept": "application/json",
    });
    req.bodyBytes = utf8.encode(jsonEncode(body));

    final streamed = await _client.send(req).timeout(timeout);
    final bodyText = await streamed.stream.transform(utf8.decoder).join().timeout(timeout);

    sw.stop();
    debugPrint("POST $url -> ${streamed.statusCode} in ${sw.elapsedMilliseconds}ms");

    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      throw Exception("HTTP ${streamed.statusCode}: $bodyText");
    }

    final trimmed = bodyText.trim();
    final decoded = jsonDecode(trimmed);
    if (decoded is Map<String, dynamic>) return decoded;

    throw Exception("Expected JSON object, got ${decoded.runtimeType}");
  }

  Future<void> _refreshHealth() async {
    try {
      final h = await _httpGetJson("$apiBase/health", timeout: healthTimeout);
      if (!mounted) return;
      setState(() => health = h);
    } catch (_) {
      if (!mounted) return;
      setState(() => health = null);
    }
  }

  void _cancelRequest() {
    if (!loading) return;
    _cancelRequested = true;

    // Closing the client aborts in-flight requests. Then recreate client.
    _client.close();
    _client = http.Client();

    setState(() {
      loading = false;
      error = "Canceled.";
    });
    _snack("Canceled");
  }

  Future<void> _analyze() async {
    setState(() {
      error = "";
      reply = "";
      meta = null;
      loading = true;
    });

    _cancelRequested = false;

    final note = noteCtrl.text.trim();
    if (note.isEmpty) {
      setState(() {
        error = "Please paste a clinical note first.";
        loading = false;
      });
      return;
    }

    try {
      final payload = {
        "preset": preset,
        "note": note,
        "debug": true,
      };

      final res = await _httpPostJsonLong(
        "$apiBase/v1/analyze",
        payload,
        timeout: analyzeTimeout,
      );

      if (_cancelRequested) return;

      final rawReply = (res["reply"] ?? "").toString();
      final cleaned = cleanModelText(rawReply);

      if (!mounted) return;
      setState(() {
        reply = cleaned;
        meta = (res["meta"] is Map<String, dynamic>) ? (res["meta"] as Map<String, dynamic>) : null;
      });

      await Future.delayed(const Duration(milliseconds: 30));
      if (replyScroll.hasClients) replyScroll.jumpTo(0);
      if (metaScroll.hasClients) metaScroll.jumpTo(0);
    } on TimeoutException {
      if (!mounted) return;
      setState(() => error =
          "Request timed out on the client. Increase analyzeTimeout or check if the API is streaming and never closing.");
      _snack("Timed out");
    } catch (e) {
      if (!mounted) return;
      setState(() => error = e.toString());
      _snack("Error");
    } finally {
      if (mounted && !_cancelRequested) setState(() => loading = false);
      _refreshHealth();
    }
  }

  Future<void> _copyText(String text) async {
    try {
      await Clipboard.setData(ClipboardData(text: text));
      _snack("Copied ✅");
    } catch (_) {}
  }

  Future<void> _logout() async {
    try {
      await FirebaseAuth.instance.signOut();
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final mq = MediaQuery.of(context);
    final isMobile = mq.size.shortestSide < 600;

    return Scaffold(
      body: Stack(
        children: [
          const _BackgroundGlow(),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 14, 14, 12),
              child: Column(
                children: [
                  _Header(
                    sweep: sweepCtrl,
                    apiReady: apiReady,
                    health: health,
                    userEmail: widget.user.email ?? "Signed in",
                    onLogout: _logout,
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: isMobile ? _mobileTabs() : _desktopSplit(),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _mobileTabs() {
    final latency = meta?["latency_ms"];
    final passes = meta?["passes"];
    final tokens = meta?["usage"] is Map ? (meta!["usage"]["total_tokens"]) : null;

    return DefaultTabController(
      length: 2,
      child: _NeonCard(
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(10, 10, 10, 6),
              decoration: BoxDecoration(
                border: Border(bottom: BorderSide(color: Colors.white.withOpacity(0.10))),
              ),
              child: Column(
                children: [
                  TabBar(
                    indicatorColor: NeonTheme.red2,
                    labelColor: Colors.white,
                    unselectedLabelColor: NeonTheme.muted,
                    dividerColor: Colors.transparent,
                    tabs: const [
                      Tab(text: "🧾 Input"),
                      Tab(text: "🫀 Output"),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 10,
                    runSpacing: 6,
                    children: [
                      Text("Preset: ", style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                      Text(preset, style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 12)),
                      Text("•", style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                      Text(presetHint, style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                    ],
                  ),
                ],
              ),
            ),
            Expanded(
              child: TabBarView(
                children: [
                  // INPUT
                  SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(12, 12, 12, 14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        _CardHeaderCompact(
                          title: "Clinical note input",
                          emoji: "🧾",
                          actions: [
                            _SmallButton(
                              label: "Load sample",
                              onTap: () => setState(() => noteCtrl.text = sampleNote),
                            ),
                            _SmallButton(
                              label: "Clear",
                              onTap: () => setState(() {
                                noteCtrl.text = "";
                                reply = "";
                                meta = null;
                                error = "";
                              }),
                            ),
                            _SmallButton(
                              label: "Cancel",
                              disabled: !loading,
                              onTap: !loading ? null : _cancelRequest,
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        SizedBox(height: 260, child: _NeonTextArea(controller: noteCtrl)),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            Expanded(
                              child: _PresetButton(
                                label: "⚡ Quick",
                                active: preset == "quick",
                                onTap: () => setState(() => preset = "quick"),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: _PresetButton(
                                label: "📋 Normal",
                                active: preset == "normal",
                                onTap: () => setState(() => preset = "normal"),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        _PresetButton(
                          label: "🧠 Detailed",
                          active: preset == "detailed",
                          onTap: () => setState(() => preset = "detailed"),
                        ),
                        const SizedBox(height: 10),
                        _PrimaryButton(
                          label: loading ? "Analyzing…" : "▶ Analyze",
                          loading: loading,
                          onTap: loading ? null : _analyze,
                        ),
                        AnimatedSwitcher(
                          duration: const Duration(milliseconds: 180),
                          child: error.isEmpty
                              ? const SizedBox(height: 8)
                              : Padding(
                                  padding: const EdgeInsets.only(top: 10),
                                  child: _Toast(text: error),
                                ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          "Reference UI for your local model. Validate clinically. 💉🩻",
                          style: TextStyle(color: NeonTheme.muted2, fontSize: 11),
                        ),
                      ],
                    ),
                  ),

                  // OUTPUT
                  SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(12, 12, 12, 14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        _CardHeaderCompact(
                          title: "Model reply",
                          emoji: "🫀",
                          actions: [
                            _SmallButton(
                              label: "Copy reply",
                              disabled: reply.isEmpty,
                              onTap: reply.isEmpty ? null : () => _copyText(reply),
                            ),
                            _SmallButton(
                              label: "Copy meta",
                              disabled: meta == null,
                              onTap: meta == null
                                  ? null
                                  : () => _copyText(const JsonEncoder.withIndent("  ").convert(meta)),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        SizedBox(
                          height: 320,
                          child: _OutputBox(
                            controller: replyScroll,
                            text: reply.isEmpty ? "No reply yet. Tap Analyze. 🩺" : reply,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Wrap(
                          spacing: 10,
                          runSpacing: 8,
                          children: [
                            Text("Meta:", style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                            _PillDim(text: latency != null ? "latency $latency ms" : "latency n/a"),
                            _PillDim(text: passes != null ? "passes $passes" : "passes n/a"),
                            _PillDim(text: tokens != null ? "tokens $tokens" : "tokens n/a"),
                          ],
                        ),
                        const SizedBox(height: 10),
                        SizedBox(
                          height: 180,
                          child: _OutputBox(
                            controller: metaScroll,
                            fill: NeonTheme.boxFill2,
                            text: meta == null
                                ? "Meta will appear here when debug=true."
                                : const JsonEncoder.withIndent("  ").convert(meta),
                            textStyle: TextStyle(color: NeonTheme.muted, fontSize: 12, height: 1.35),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _desktopSplit() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final isNarrow = constraints.maxWidth < 980;
        return isNarrow
            ? Column(
                children: [
                  Expanded(child: _leftCard()),
                  const SizedBox(height: 14),
                  Expanded(child: _rightCard()),
                ],
              )
            : Row(
                children: [
                  Expanded(flex: 11, child: _leftCard()),
                  const SizedBox(width: 14),
                  Expanded(flex: 9, child: _rightCard()),
                ],
              );
      },
    );
  }

  Widget _leftCard() {
    return _NeonCard(
      child: Column(
        children: [
          _CardHeader(
            title: "Clinical note input",
            emoji: "🧾",
            actions: [
              _SmallButton(
                label: "Load sample",
                onTap: () => setState(() => noteCtrl.text = sampleNote),
              ),
              _SmallButton(
                label: "Clear",
                onTap: () => setState(() {
                  noteCtrl.text = "";
                  reply = "";
                  meta = null;
                  error = "";
                }),
              ),
              _SmallButton(
                label: "Cancel",
                disabled: !loading,
                onTap: !loading ? null : _cancelRequest,
              ),
            ],
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
              child: Column(
                children: [
                  Expanded(child: _NeonTextArea(controller: noteCtrl)),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 10,
                    runSpacing: 6,
                    children: [
                      Text("Preset:", style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                      Text(preset, style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 12)),
                      Text("•", style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                      Text(presetHint, style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: _PresetButton(
                          label: "⚡ Quick",
                          active: preset == "quick",
                          onTap: () => setState(() => preset = "quick"),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: _PresetButton(
                          label: "📋 Normal",
                          active: preset == "normal",
                          onTap: () => setState(() => preset = "normal"),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: _PresetButton(
                          label: "🧠 Detailed",
                          active: preset == "detailed",
                          onTap: () => setState(() => preset = "detailed"),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  _PrimaryButton(
                    label: loading ? "Analyzing…" : "▶ Analyze",
                    loading: loading,
                    onTap: loading ? null : _analyze,
                  ),
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 180),
                    child: error.isEmpty
                        ? const SizedBox(height: 8)
                        : Padding(
                            padding: const EdgeInsets.only(top: 10),
                            child: _Toast(text: error),
                          ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    "Reference UI for your local model. Validate clinically. 💉🩻",
                    style: TextStyle(color: NeonTheme.muted2, fontSize: 11),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _rightCard() {
    final latency = meta?["latency_ms"];
    final passes = meta?["passes"];
    final tokens = meta?["usage"] is Map ? (meta!["usage"]["total_tokens"]) : null;

    return _NeonCard(
      child: Column(
        children: [
          _CardHeader(
            title: "Model reply",
            emoji: "🫀",
            actions: [
              _SmallButton(
                label: "Copy reply",
                disabled: reply.isEmpty,
                onTap: reply.isEmpty ? null : () => _copyText(reply),
              ),
              _SmallButton(
                label: "Copy meta",
                disabled: meta == null,
                onTap: meta == null ? null : () => _copyText(const JsonEncoder.withIndent("  ").convert(meta)),
              ),
            ],
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(14, 12, 14, 14),
              child: Column(
                children: [
                  Expanded(
                    child: _OutputBox(
                      controller: replyScroll,
                      text: reply.isEmpty ? "No reply yet. Tap Analyze. 🩺" : reply,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 10,
                    runSpacing: 8,
                    children: [
                      Text("Meta:", style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
                      _PillDim(text: latency != null ? "latency $latency ms" : "latency n/a"),
                      _PillDim(text: passes != null ? "passes $passes" : "passes n/a"),
                      _PillDim(text: tokens != null ? "tokens $tokens" : "tokens n/a"),
                    ],
                  ),
                  const SizedBox(height: 10),
                  SizedBox(
                    height: 170,
                    child: _OutputBox(
                      controller: metaScroll,
                      fill: NeonTheme.boxFill2,
                      text: meta == null
                          ? "Meta will appear here when debug=true."
                          : const JsonEncoder.withIndent("  ").convert(meta),
                      textStyle: TextStyle(color: NeonTheme.muted, fontSize: 12, height: 1.35),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/* ------------------ background ------------------ */

class _BackgroundGlow extends StatelessWidget {
  const _BackgroundGlow();

  @override
  Widget build(BuildContext context) {
    return Container(
      color: NeonTheme.bg,
      child: Stack(
        children: [
          Positioned(left: -180, top: -120, child: _radial(520, NeonTheme.red.withOpacity(0.16))),
          Positioned(right: -200, bottom: -160, child: _radial(560, NeonTheme.red.withOpacity(0.10))),
          Positioned.fill(
            child: IgnorePointer(
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [Colors.white.withOpacity(0.02), Colors.transparent],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _radial(double size, Color c) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(colors: [c, Colors.transparent], stops: const [0.0, 0.70]),
      ),
    );
  }
}

/* ------------------ header ------------------ */

class _Header extends StatelessWidget {
  const _Header({
    required this.sweep,
    required this.apiReady,
    required this.health,
    required this.userEmail,
    required this.onLogout,
  });

  final Animation<double> sweep;
  final bool apiReady;
  final Map<String, dynamic>? health;
  final String userEmail;
  final VoidCallback onLogout;

  double _lerp(double a, double b, double t) => a + (b - a) * t;

  @override
  Widget build(BuildContext context) {
    final w = MediaQuery.of(context).size.width;
    final compact = w < 560;

    final gpu = (health?["gpu"] ?? "").toString();
    final q4 = health?["is_loaded_in_4bit"];

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              border: Border.all(color: NeonTheme.red.withOpacity(0.35)),
              borderRadius: BorderRadius.circular(16),
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Colors.white.withOpacity(0.05), Colors.white.withOpacity(0.03)],
              ),
            ),
            padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
            child: compact
                ? Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          _Logo(),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  "Triage Assist-SA:",
                                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900),
                                ),
                                const SizedBox(height: 3),
                                Text(
                                  "AI Clinical Decision Support For ICU Patient Care 🫀",
                                  style: TextStyle(color: NeonTheme.muted, fontSize: 12),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Wrap(
                        spacing: 10,
                        runSpacing: 8,
                        children: [
                          _Pill(text: apiReady ? "API: Ready" : "API: Offline", ok: apiReady),
                          _Pill(text: gpu.isNotEmpty ? "GPU: $gpu" : "GPU: n/a"),
                          _Pill(text: q4 == true ? "4-bit: Yes" : (q4 == false ? "4-bit: No" : "4-bit: ?")),
                          _PillDim(text: userEmail),
                          _SmallButton(label: "Logout", onTap: onLogout),
                        ],
                      ),
                    ],
                  )
                : Row(
                    children: [
                      _Logo(),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              "TriageAssist Clinical Console",
                              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                            ),
                            const SizedBox(height: 3),
                            Text(
                              "Local inference • Quick / Normal / Detailed • Neon Red Theme 🫀",
                              style: TextStyle(color: NeonTheme.muted, fontSize: 12),
                            ),
                            const SizedBox(height: 8),
                            Wrap(
                              spacing: 10,
                              runSpacing: 8,
                              children: [
                                _PillDim(text: userEmail),
                                _SmallButton(label: "Logout", onTap: onLogout),
                              ],
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          _Pill(text: apiReady ? "API: Ready" : "API: Offline", ok: apiReady),
                          const SizedBox(height: 8),
                          _Pill(text: gpu.isNotEmpty ? "GPU: $gpu" : "GPU: n/a"),
                          const SizedBox(height: 8),
                          _Pill(text: q4 == true ? "4-bit: Yes" : (q4 == false ? "4-bit: No" : "4-bit: ?")),
                        ],
                      ),
                    ],
                  ),
          ),
          Positioned.fill(
            child: IgnorePointer(
              child: AnimatedBuilder(
                animation: sweep,
                builder: (context, _) {
                  final t = sweep.value;
                  final dx = _lerp(-1.2, 1.2, t);
                  return Transform.translate(
                    offset: Offset(dx * MediaQuery.of(context).size.width, 0),
                    child: Opacity(
                      opacity: 0.20,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              Colors.transparent,
                              NeonTheme.red.withOpacity(0.22),
                              Colors.transparent,
                            ],
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Logo extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: NeonTheme.red.withOpacity(0.35)),
        boxShadow: [NeonTheme.glow()],
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [NeonTheme.red.withOpacity(0.18), Colors.white.withOpacity(0.02)],
        ),
      ),
      alignment: Alignment.center,
      child: const Text("🩺", style: TextStyle(fontSize: 20)),
    );
  }
}

/* ------------------ shared widgets ------------------ */

class _NeonCard extends StatelessWidget {
  const _NeonCard({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withOpacity(0.12)),
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Colors.white.withOpacity(0.05), Colors.white.withOpacity(0.03)],
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: child,
    );
  }
}

class _CardHeader extends StatelessWidget {
  const _CardHeader({required this.title, required this.emoji, required this.actions});
  final String title;
  final String emoji;
  final List<Widget> actions;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: Colors.white.withOpacity(0.10))),
      ),
      child: Row(
        children: [
          Text(
            "$emoji  $title",
            style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 13, letterSpacing: 0.2),
          ),
          const Spacer(),
          Wrap(spacing: 10, runSpacing: 8, children: actions),
        ],
      ),
    );
  }
}

class _CardHeaderCompact extends StatelessWidget {
  const _CardHeaderCompact({required this.title, required this.emoji, required this.actions});
  final String title;
  final String emoji;
  final List<Widget> actions;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(10, 10, 10, 10),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.10)),
        color: Colors.white.withOpacity(0.03),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "$emoji  $title",
            style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 13, letterSpacing: 0.2),
          ),
          const SizedBox(height: 8),
          Wrap(spacing: 10, runSpacing: 8, children: actions),
        ],
      ),
    );
  }
}

class _OutputBox extends StatelessWidget {
  const _OutputBox({
    required this.controller,
    required this.text,
    this.fill,
    this.textStyle,
  });

  final ScrollController controller;
  final String text;
  final Color? fill;
  final TextStyle? textStyle;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: fill ?? NeonTheme.boxFill,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: NeonTheme.red.withOpacity(0.26)),
      ),
      clipBehavior: Clip.antiAlias,
      padding: const EdgeInsets.all(12),
      child: Scrollbar(
        controller: controller,
        child: SingleChildScrollView(
          controller: controller,
          child: Text(text, style: textStyle ?? const TextStyle(height: 1.35)),
        ),
      ),
    );
  }
}

class _NeonTextArea extends StatefulWidget {
  const _NeonTextArea({required this.controller});
  final TextEditingController controller;

  @override
  State<_NeonTextArea> createState() => _NeonTextAreaState();
}

class _NeonTextAreaState extends State<_NeonTextArea> {
  final FocusNode focus = FocusNode();

  @override
  void dispose() {
    focus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final focused = focus.hasFocus;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 150),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: NeonTheme.red.withOpacity(focused ? 0.70 : 0.26)),
        color: NeonTheme.boxFill,
        boxShadow: focused ? [NeonTheme.glow()] : [],
      ),
      clipBehavior: Clip.antiAlias,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      child: TextField(
        controller: widget.controller,
        focusNode: focus,
        keyboardType: TextInputType.multiline,
        maxLines: null,
        expands: true,
        style: const TextStyle(color: NeonTheme.text, height: 1.35),
        decoration: const InputDecoration(
          border: InputBorder.none,
          hintText: "Paste or type the clinical note / case details here…",
          hintStyle: TextStyle(color: Color(0x80FFFFFF)),
        ),
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  const _Pill({required this.text, this.ok});
  final String text;
  final bool? ok;

  @override
  Widget build(BuildContext context) {
    Color border = Colors.white.withOpacity(0.14);
    Color fg = NeonTheme.muted;

    if (ok == true) {
      border = const Color(0x4000FFA0);
      fg = Colors.white;
    } else if (ok == false) {
      border = NeonTheme.red.withOpacity(0.35);
      fg = const Color(0xFFFFD2D2);
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: border),
        color: Colors.white.withOpacity(0.04),
      ),
      child: Text(text, style: TextStyle(color: fg, fontSize: 12)),
    );
  }
}

class _PillDim extends StatelessWidget {
  const _PillDim({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
        color: Colors.white.withOpacity(0.04),
      ),
      child: Text(text, style: TextStyle(color: NeonTheme.muted, fontSize: 12)),
    );
  }
}

class _SmallButton extends StatelessWidget {
  const _SmallButton({required this.label, required this.onTap, this.disabled = false});
  final String label;
  final VoidCallback? onTap;
  final bool disabled;

  @override
  Widget build(BuildContext context) {
    final canTap = !disabled && onTap != null;

    return InkWell(
      onTap: canTap ? onTap : null,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white.withOpacity(0.14)),
          color: Colors.white.withOpacity(0.04),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontWeight: FontWeight.w800,
            fontSize: 12,
            color: canTap ? NeonTheme.text : NeonTheme.muted2,
          ),
        ),
      ),
    );
  }
}

class _PresetButton extends StatefulWidget {
  const _PresetButton({required this.label, required this.active, required this.onTap});
  final String label;
  final bool active;
  final VoidCallback onTap;

  @override
  State<_PresetButton> createState() => _PresetButtonState();
}

class _PresetButtonState extends State<_PresetButton> {
  bool pressed = false;

  @override
  Widget build(BuildContext context) {
    final active = widget.active;

    return GestureDetector(
      onTapDown: (_) => setState(() => pressed = true),
      onTapCancel: () => setState(() => pressed = false),
      onTapUp: (_) => setState(() => pressed = false),
      onTap: widget.onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        transform: Matrix4.identity()..translate(0.0, pressed ? 1.0 : 0.0),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: NeonTheme.red.withOpacity(active ? 0.70 : 0.14)),
          color: active ? NeonTheme.red.withOpacity(0.12) : Colors.white.withOpacity(0.04),
          boxShadow: active ? [NeonTheme.glow()] : [],
        ),
        padding: const EdgeInsets.symmetric(vertical: 12),
        alignment: Alignment.center,
        child: Text(widget.label, style: const TextStyle(fontWeight: FontWeight.w900, fontSize: 12)),
      ),
    );
  }
}

class _PrimaryButton extends StatefulWidget {
  const _PrimaryButton({required this.label, required this.onTap, required this.loading});
  final String label;
  final VoidCallback? onTap;
  final bool loading;

  @override
  State<_PrimaryButton> createState() => _PrimaryButtonState();
}

class _PrimaryButtonState extends State<_PrimaryButton> {
  bool pressed = false;

  @override
  Widget build(BuildContext context) {
    final canTap = widget.onTap != null;
    final hot = widget.loading;

    final borderOpacity = hot ? 0.85 : 0.60;
    final top = hot ? 0.30 : 0.22;
    final bot = hot ? 0.16 : 0.10;

    return GestureDetector(
      onTapDown: (_) => setState(() => pressed = true),
      onTapCancel: () => setState(() => pressed = false),
      onTapUp: (_) => setState(() => pressed = false),
      onTap: widget.onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        transform: Matrix4.identity()..translate(0.0, pressed ? 1.0 : 0.0),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: NeonTheme.red.withOpacity(borderOpacity)),
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              NeonTheme.red.withOpacity(top),
              NeonTheme.red.withOpacity(bot),
            ],
          ),
          boxShadow: canTap ? [hot ? NeonTheme.glowStrong() : NeonTheme.glow(strength: 0.85)] : [],
        ),
        padding: const EdgeInsets.symmetric(vertical: 14),
        alignment: Alignment.center,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (widget.loading) ...[
              SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white.withOpacity(0.9)),
                ),
              ),
              const SizedBox(width: 10),
            ],
            Text(
              widget.label,
              style: TextStyle(
                fontWeight: FontWeight.w900,
                fontSize: 13,
                color: canTap ? NeonTheme.text : NeonTheme.muted2,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Toast extends StatelessWidget {
  const _Toast({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: NeonTheme.red.withOpacity(0.35)),
        color: NeonTheme.red.withOpacity(0.08),
        boxShadow: [NeonTheme.glow()],
      ),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      child: Text("⚠ $text", style: const TextStyle(color: Color(0xFFFFD8D8))),
    );
  }
}
