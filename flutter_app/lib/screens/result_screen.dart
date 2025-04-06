import 'package:flutter/material.dart';
import 'package:flutter_app/models/waste_classification.dart';

class ResultScreen extends StatelessWidget {
  final WasteClassification result;

  ResultScreen({required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Kết quả phân loại'),
        backgroundColor:
            Colors.green[700], // Màu nền AppBar đồng bộ với HomeScreen
        elevation: 4,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.green[50]!, Colors.white], // Gradient nền
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Card chứa thông tin chính
                Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildResultItem(
                          icon: Icons.delete,
                          title: 'Dự đoán',
                          value: result.className,
                        ),
                        const Divider(),
                        _buildResultItem(
                          icon: Icons.check_circle,
                          title: 'Độ tin cậy',
                          value:
                              '${(result.confidence * 100).toStringAsFixed(2)}%',
                        ),
                        const Divider(),
                        _buildResultItem(
                          icon: Icons.category,
                          title: 'Lớp',
                          value: result.category,
                        ),
                        const Divider(),
                        _buildResultItem(
                          icon: Icons.info,
                          title: 'Huớng dẫn xử lý',
                          value: result.disposalIntruction,
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                // Tiêu đề Top Predictions
                const Text(
                  'Top 5 dự đoán',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.green,
                  ),
                ),
                const SizedBox(height: 10),
                // Danh sách Top Predictions trong Card
                Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children:
                          result.topPredictions.asMap().entries.map((entry) {
                            final index = entry.key;
                            final pred = entry.value;

                            final colors = [
                              Colors.red, // Top 1
                              Colors.orange, // Top 2
                              Colors.yellow[700]!, // Top 3
                              Colors.lightBlue, // Top 4
                              Colors.lightGreen, // Top 5
                            ];

                            final textColor =
                                index < colors.length
                                    ? colors[index]
                                    : Colors.lightGreen;

                            return Padding(
                              padding: const EdgeInsets.symmetric(
                                vertical: 4.0,
                              ),
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    pred['class'],
                                    style: const TextStyle(fontSize: 16),
                                  ),
                                  Text(
                                    '${(pred['confidence'] * 100).toStringAsFixed(2)}%',
                                    style: TextStyle(
                                      fontSize: 16,
                                      color: textColor,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                            );
                          }).toList(),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// Hàm hỗ trợ tạo item kết quả với icon
Widget _buildResultItem({
  required IconData icon,
  required String title,
  required String value,
}) {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 6.0),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: Colors.green[600], size: 24),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                value,
                style: TextStyle(fontSize: 18, color: Colors.black),
              ),
            ],
          ),
        ),
      ],
    ),
  );
}
