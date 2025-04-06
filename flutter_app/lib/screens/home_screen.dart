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
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => ResultScreen(result: result)),
      );
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('UrlSubmitted Error: ${e}')));
      print(e);
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Waste Classification'),
        backgroundColor: Colors.green[700],
        elevation: 4,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.green[50]!, Colors.white],
          ),
        ),
        child: Center(
          child:
              _isLoading
                  ? CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(Colors.green),
                  )
                  : Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Text(
                          'Model: EfficientNet-B0',
                          style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: Colors.green,
                          ),
                        ),
						const SizedBox(height: 40),
                        ImagePickerWidget(onImagePicked: _handleImagePicked),
                        SizedBox(height: 25),
                        UrlInputWidget(onUrlSubmitted: _handleUrlSubmitted),
                      ],
                    ),
                  ),
        ),
      ),
    );
  }
}
