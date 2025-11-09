!pip install streamlit #install dependencies
!pip install torch
!pip install pillow

!wget -q -O - ipv4.icanhazip.com # tunnel code

!streamlit run /content/drive/MyDrive/SolarFilamentDetection/program.py & npx localtunnel --port 8501 #implementation