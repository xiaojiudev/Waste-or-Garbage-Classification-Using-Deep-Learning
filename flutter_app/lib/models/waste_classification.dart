class WasteClassification {
	final String className;
	final double confidence;
	final String category;
	final String disposalIntruction;
	final List<Map<String, dynamic>> topPredictions;

	WasteClassification({
		required this.className,
		required this.confidence,
		required this.category,
		required this.disposalIntruction,
		required this.topPredictions,
	});

	factory WasteClassification.fromJson(Map<String, dynamic> json) {
		return WasteClassification(
			className: json['class'] ?? '',
			confidence: json['confidence'].toDouble(),
			category: json['category'] ?? '',
			disposalIntruction: json['disposal_instruction'] ?? '',
			topPredictions: List<Map<String, dynamic>>.from(json['top_predictions']),
		);
	}
}