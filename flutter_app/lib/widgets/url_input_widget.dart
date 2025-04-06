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
          decoration: InputDecoration(
            labelText: 'Enter image URL',
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            focusedBorder: OutlineInputBorder(
              borderSide: const BorderSide(color: Colors.green, width: 2),
              borderRadius: BorderRadius.circular(12),
            ),
            labelStyle: const TextStyle(color: Colors.green),
          ),
		  keyboardType: TextInputType.url,
        ),
		const SizedBox(height: 20),
        ElevatedButton.icon(
          onPressed: () => widget.onUrlSubmitted(_controller.text),
          icon: const Icon(Icons.cloud_upload),
          label: const Text('Classify from URL'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.green[600],
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
            textStyle: const TextStyle(fontSize: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ],
    );
  }
}
