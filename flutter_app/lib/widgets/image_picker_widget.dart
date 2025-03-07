import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class ImagePickerWidget extends StatelessWidget {
  final Function(File) onImagePicked;

  ImagePickerWidget({required this.onImagePicked});

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      onImagePicked(File(pickedFile.path));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ElevatedButton(
          onPressed: () => _pickImage(ImageSource.camera),
          child: Text('Take a photo'),
        ),
        ElevatedButton(
          onPressed: () => _pickImage(ImageSource.gallery),
          child: Text('Select from gallery'),
        ),
      ],
    );
  }
}
