// import 'package:flutter/material.dart';
// import 'package:provider/provider.dart';

// class HomeScreen extends StatelessWidget {
//   const HomeScreen({super.key});

//   @override
//   Widget build(BuildContext context) {
//     final viewModel = Provider.of<HomeViewModel>(context);

//     return Scaffold(
//       appBar: AppBar(title: const Text('Phân loại rác thải')),
//       body: SingleChildScrollView(
//         padding: const EdgeInsets.all(16),
//         child: Column(
//           children: [
//             _buildImagePreview(viewModel),
//             const SizedBox(height: 20),
//             _buildImageSourceButtons(viewModel),
//             const SizedBox(height: 20),
//             _buildUrlInputField(viewModel),
//             const SizedBox(height: 30),
//             _buildSubmitButton(viewModel),
//           ],
//         ),
//       ),
//     );
//   }
// }
