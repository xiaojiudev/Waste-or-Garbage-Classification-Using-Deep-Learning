import 'package:flutter/material.dart';
import 'package:flutter_app/models/waste_classification.dart';
import 'package:google_fonts/google_fonts.dart';

class ResultScreen extends StatelessWidget {
  final WasteClassification result;

  ResultScreen({required this.result});


  @override
  Widget build(BuildContext context) {
    return Scaffold(
		appBar: AppBar(title: Text('Classification Result')),
		body: Padding(
			padding: const EdgeInsets.all(16.0),
			child: Column(
				crossAxisAlignment: CrossAxisAlignment.start,
				children: [
					Text('Waste Type: ${result.className}', style: TextStyle(fontSize: 18),),
					Text('Confidence: ${(result.confidence * 100).toStringAsFixed(3)}%', style: TextStyle(fontSize: 18),),
					Text('Category: ${result.category}', style: TextStyle(fontSize: 18),),
					Text('Disposal Instructions: ${result.disposalIntruction}', style: GoogleFonts.notoSans(fontSize: 18),),
					SizedBox(height: 20,),
					Text('Top predictions:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
					...result.topPredictions.map((pred) => Text(
						'${pred['class']}: ${(pred['confidence'] * 100).toStringAsFixed(3)}%')),
				],
			),
		),

	);

  }
}
