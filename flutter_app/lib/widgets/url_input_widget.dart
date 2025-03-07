import 'package:flutter/material.dart';

class UrlInputWidget extends StatefulWidget {
	final Function(String) onUrlSubmitted;

	UrlInputWidget({required this.onUrlSubmitted});

	@override
	_UrlInputWidgetState createState() => _UrlInputWidgetState();

}

class _UrlInputWidgetState extends State<UrlInputWidget> {
	final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
	return Column(
		children: [
			TextField(
				controller: _controller,
				decoration: InputDecoration(labelText: 'Enter image URL'),
			),
			ElevatedButton(
				onPressed: () => widget.onUrlSubmitted(_controller.text),
				child: Text('Classify from URL'),
			)
		],
	);
  }
}
