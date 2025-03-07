import 'dart:io';
import 'package:flutter/material.dart';

import 'package:flutter_app/services/api_service.dart';
import 'package:flutter_app/screens/result_screen.dart';
import 'package:flutter_app/widgets/image_picker_widget.dart';
import 'package:flutter_app/widgets/url_input_widget.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _apiService = ApiService();
  bool _isLoading = false;

  void _handleImagePicked(File image) async {
    setState(() => _isLoading = true);
    try {
      final result = await _apiService.classifyImage(image);
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => ResultScreen(result: result)),
      );
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('ImagePicked Error: ${e}')));
	  print(e);
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _handleUrlSubmitted(String url) async {
	setState(() => _isLoading = true);
	try {
	  final result = await _apiService.classifyImageUrl(url);
	  Navigator.push(context, MaterialPageRoute(builder: (context) => ResultScreen(result: result)));
	} catch (e) {
	  ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('UrlSubmitted Error: ${e}')));
	  print(e);
	} finally {
		setState(() => _isLoading = false);
	}
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
		appBar: AppBar(title: Text('Waste Classification')),
		body: Center(
			child: _isLoading 
			? CircularProgressIndicator()
			: Column(
				mainAxisAlignment: MainAxisAlignment.center,
				children: [
					ImagePickerWidget(onImagePicked: _handleImagePicked,),
					SizedBox(height: 20,),
					UrlInputWidget(onUrlSubmitted: _handleUrlSubmitted,),
				],
			),
		),

	);
  }
}
