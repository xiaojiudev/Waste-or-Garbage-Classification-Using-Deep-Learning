import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
	static final ThemeData lightTheme = ThemeData(
		primarySwatch: Colors.blue,
		visualDensity: VisualDensity.adaptivePlatformDensity,
		textTheme: GoogleFonts.notoSansTextTheme(),
	);
}