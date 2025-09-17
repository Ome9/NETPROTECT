# NetProtect AI Network Anomaly Detection Suite

🛡️ **Live Network Anomaly Detection Dashboard** using React, Node.js, and TypeScript with real-time monitoring capabilities powered by advanced AI/ML models.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+ with PyTorch
- Git LFS (for model files)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NetProtect
   git lfs pull  # Download AI models
   ```

2. **Download Datasets** (Required for training)
   
   The following datasets are **not included** in the repository due to size. Download them separately:
   
   - **NSL-KDD Dataset**: Place in `NSL-KDD_Dataset/`
     - Download from: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
     - Required files: `KDDTrain+.txt`, `KDDTest+.txt`, `KDDTest-21.txt`
   
   - **CSE-CIC-IDS2018**: Place in `CSE-CIC_Dataset/`  
     - Download from: [CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
     - Required files: `*.parquet` files for each day
   
   - **UNSW-NB15**: Place in `UNSW_Dataset/`
     - Download from: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
     - Required files: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`

3. **Install Dashboard Dependencies**
   ```bash
   cd network-anomaly-dashboard
   npm install
   ```

4. **Setup Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Dashboard**
   ```bash
   cd network-anomaly-dashboard
   npm run dev
   ```

2. **Run Production Model**
   ```bash
   python unified_production_model.py
   ```

## 📁 Project Structure

```
NetProtect/
├── 🎯 network-anomaly-dashboard/     # Next.js React Dashboard
├── 🤖 models/                       # AI/ML Models (via Git LFS)
│   ├── production/                  # Production-ready models
│   ├── preprocessors/               # Data preprocessing models  
│   └── model_registry.json          # Model metadata
├── 📊 results/                      # Training results & metrics
├── 🔧 src/                         # Training scripts & utilities
├── 📈 plots/                       # Visualization outputs
├── 🎨 UIUX/                        # UI/UX design assets
└── 📚 Datasets/                    # (Download separately)
    ├── NSL-KDD_Dataset/
    ├── CSE-CIC_Dataset/
    └── UNSW_Dataset/
```

## 🧠 AI Models

The suite includes multiple pre-trained models optimized for different scenarios:

- **VAE Autoencoder** (95.17% accuracy) - Primary production model
- **Baseline Autoencoder** (92.5% accuracy) - Standard baseline
- **Improved GPU Model** (90.03% accuracy) - GPU-optimized for CSE-CIC
- **Max GPU Model** (91.84% accuracy) - Maximum GPU utilization for UNSW-NB15

## 🛠️ Technology Stack

- **Frontend**: React 18, Next.js 15, TypeScript, Tailwind CSS
- **Backend**: Node.js, Express, Socket.io WebSocket
- **AI/ML**: PyTorch, Scikit-learn, Pandas, NumPy  
- **UI Components**: Shadcn/ui, Recharts, Framer Motion
- **Real-time**: Socket.io for live data streaming

## 🔐 Security Features

- Real-time network traffic monitoring
- Multi-dataset anomaly detection (NSL-KDD, CIC-IDS2018, UNSW-NB15)
- Live model inference with confidence scoring
- Network adapter monitoring and log forwarding
- Interactive dashboards with alerts and notifications

## 📖 Documentation

- [`FINAL_PROJECT_REPORT.md`](FINAL_PROJECT_REPORT.md) - Complete project documentation
- [`models/model_registry.json`](models/model_registry.json) - Model metadata and specifications
- [`network-anomaly-dashboard/README.md`](network-anomaly-dashboard/README.md) - Dashboard setup guide

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support, please refer to the documentation or open an issue in the repository.

---

**⚠️ Important**: Remember to download the required datasets separately as they are not included in this repository due to size constraints.