# 🛡️ Complete Air-Gapped Deployment Guide for AI Appointer Assist

This guide details exactly how to deploy **AI Appointer Assist** in a high-security, **no-internet (air-gapped)** environment.

## 📋 Pre-Deployment Checklist

Before you begin the transfer, ensure you have the following on your **Internet-Connected (Packager)** machine:

- [x] **Source Code**: The `AIAppointer` folder is clean and structured.
- [x] **Data**: `data/hr_star_trek_v4c_modernized_clean_modified_v4.csv` is present.
- [x] **Models**: Trained models (`lgbm_model.pkl`, etc.) are in `models/`.
- [ ] **Dependencies (CRITICAL)**: You must download Python packages (wheels) now.

---

## 📦 Phase 1: Preparation & Packaging (Internet Zone)

Perform these steps on a machine with internet access.

### 1. Download Offline Dependencies (The "wheels")
You need to download the exact Python libraries required so they can be installed without internet.

**Option A: Target Server is the SAME OS (e.g., Windows → Windows)**
```powershell
mkdir wheels
pip download -r requirements.txt -d wheels/
```

**Option B: Target Server is LINUX (e.g., Windows → Red Hat/Ubuntu)**
If you are packaging on Windows but deploying to Linux, you must specify the platform:
```powershell
mkdir wheels
pip download -r requirements.txt -d wheels/ --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.9
```

### 2. Verify Python Installer
Does your air-gapped server already have Python 3.8+ installed?
- **YES**: Skip this step.
- **NO**: Download the Python installer (Windows installer or Linux source tarball) and place it in a new `python_installer/` folder.

### 3. Create the Deployment Archive
Compress everything into a single file for easy transfer.

**Windows (PowerShell)**:
```powershell
Compress-Archive -Path AIAppointer, wheels -DestinationPath AIAppointer_Deploy_v1.0.zip
```

**Linux/Mac**:
```bash
tar -czf AIAppointer_Deploy_v1.0.tar.gz AIAppointer/ wheels/
```

---

## 🚚 Phase 2: The Shift (Transfer)

Move the `AIAppointer_Deploy_v1.0` archive to your secure network.
- **Method**: Secure USB, CD/DVD, or One-Way Data Transfer Gateway.
- **Scan**: Perform required virus/malware scans according to your organization's policy.

---

## 🔒 Phase 3: Installation (Secure Zone)

Perform these steps on the **Air-Gapped Target Server**.

### 1. Extract Files
Unzip the package to your desired location (e.g., `C:\Apps\` or `/opt/`).

**Windows**:
Right-click > Extract All... to `C:\Apps\AIAppointer`. Ensure the `wheels` folder is inside or next to it.

**Linux**:
```bash
tar -xzf AIAppointer_Deploy_v1.0.tar.gz -C /opt/
```

### 2. Install Python (If Missing)
Run the Python installer you brought in Phase 1 if Python 3.8+ is not known to the system.

### 3. Run the Installation Script
This script creates a virtual environment and installs the dependencies from the `wheels/` folder.

**Windows (PowerShell)**:
```powershell
cd C:\Apps\AIAppointer
# Ensure 'wheels' folder is present in root of AIAppointer
.\scripts\install.sh 
# (Note: You may need to run this in Git Bash or use standard pip commands if .sh is not supported)
```

**Windows (Manual Alternative)**:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --no-index --find-links=../wheels -r requirements.txt
```

**Linux**:
```bash
cd /opt/AIAppointer
chmod +x scripts/install.sh
./scripts/install.sh
```

---

## 🚀 Phase 4: Server Deployment

### 1. Configure the Network
To allow other users on the local network to access the app, you need to know the server's IP address.

**Find IP Address**:
- Windows: `ipconfig` (Look for IPv4 Address, e.g., `192.168.1.50`)
- Linux: `ip a` (Look for inet address)

### 2. Start the Application
Run the start script. It is configured to listen on `0.0.0.0` (all network interfaces).

**Windows**:
```powershell
.\venv\Scripts\activate
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

**Linux**:
```bash
./scripts/start.sh
```

### 3. Configure Firewall (Crucial!)
The server usually blocks incoming connections by default. You must allow traffic on port **8501**.

**Windows Firewall**:
1. Open "Windows Defender Firewall with Advanced Security".
2. Inbound Rules > New Rule.
3. Port > TCP > Specific local ports: **8501**.
4. Allow the connection.
5. Name: "AI Appointer Web".

**Linux (UFW)**:
```bash
sudo ufw allow 8501/tcp
```

---

## 🌐 Phase 5: End User Access

Users on the secure local network can now access the application.

**Instructions for Users**:
1. Open a modern web browser (Edge, Chrome, Firefox).
2. Navigate to: `http://<SERVER_IP>:8501`
   - Example: `http://192.168.1.50:8501`

### Troubleshooting Connectivity
- **"Site can't be reached"**: Check Server Firewall (Phase 4, Step 3) and ensure Server and Client are on the same subnet/VLAN.
- **Application Error**: Check server logs in the terminal window.

---

## 🗄️ Database & Model Maintenance
Since there is no internet, the app will **not** auto-update.

- **Data Updates**: Replace `data/hr_...csv` manually.
- **Retraining**: Run `python scripts/train_model.py` on the server after updating data.
