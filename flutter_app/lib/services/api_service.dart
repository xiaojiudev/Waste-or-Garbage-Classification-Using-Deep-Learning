import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

import 'package:flutter_app/utils/constants.dart';
import 'package:flutter_app/models/waste_classification.dart';
import 'package:http_parser/http_parser.dart';

class ApiService {
  // Send image from file
  Future<WasteClassification> classifyImage(File image) async {
    final uri = Uri.parse('${Constants.apiUrl}/predict/file');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath(
          'item',
          image.path,
          contentType: MediaType('image', 'jpeg'),
        ),
      );
    final response = await request.send();

    if (response.statusCode == 200) {
      final responseData = await response.stream.bytesToString();
      return WasteClassification.fromJson(jsonDecode(responseData));
    } else {
      final errorData = await response.stream.bytesToString();
      print('Error response: $errorData');
      throw Exception('Cannot classify image: ${response.statusCode}');
    }
  }

  // Send image from URL
  Future<WasteClassification> classifyImageUrl(String url) async {
    final uri = Uri.parse('${Constants.apiUrl}/predict/url');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url}),
    );

    if (response.statusCode == 200) {
      final utf8Decoded = utf8.decode(response.bodyBytes);
      return WasteClassification.fromJson(jsonDecode(utf8Decoded));
    } else {
      final errorData = response.body;
      print('Error response: $errorData');
      throw Exception('Cannot classify image from URL: ${response.statusCode}');
    }
  }
}
