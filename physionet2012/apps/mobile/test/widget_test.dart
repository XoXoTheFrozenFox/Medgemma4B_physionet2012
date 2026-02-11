// test/widget_test.dart
//
// Basic widget smoke test for the current MedGemma UI.
// (Replaces the default counter test.)

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:llm_medgemma/main.dart';

void main() {
  testWidgets('App builds (smoke test)', (WidgetTester tester) async {
    // If your MedGemmaApp now requires firebaseInitError, pass a safe value.
    // If your constructor does NOT require it, you can remove the argument.
    await tester.pumpWidget(const MedGemmaApp(firebaseInitError: ""));

    // Let first frame settle (animations/timers may still be running, so don't pumpAndSettle).
    await tester.pump(const Duration(milliseconds: 50));

    // Basic assertions that the app rendered.
    expect(find.byType(MaterialApp), findsOneWidget);

    // Optional: check for a known title string from your UI (adjust if you renamed it).
    expect(find.textContaining('Triage'), findsWidgets);
  });
}
