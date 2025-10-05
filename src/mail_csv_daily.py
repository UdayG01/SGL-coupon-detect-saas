import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import schedule
import time
import os

# -------------------
# Email configuration
# -------------------
SENDER_EMAIL = "uday.renataiot@gmail.com"     # Replace with your email
APP_PASSWORD = "gwpc fwfi lnrw pphm"   # Replace with generated app password
RECEIVER_EMAIL = "guptauday49@gmail.com"  # Replace with recipient email
CSV_FILE = "src/results.csv"                  # Path to your CSV file

def send_email_daily_log():
    """Send an email with CSV attachment."""
    msg = MIMEMultipart()
    msg["Subject"] = "Automated Report with CSV"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    # Email body
    body = "Hello! This is an automated email with a CSV attachment."
    msg.attach(MIMEText(body, "plain"))

    # Attach CSV if it exists
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(CSV_FILE)}")
        msg.attach(part)
    else:
        print(f"⚠️ CSV file '{CSV_FILE}' not found, sending email without attachment.")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("✅ Email with CSV sent successfully")
    except Exception as e:
        print("❌ Error sending email:", e)

# -------------------
# Schedule job
# -------------------
schedule.every(30).seconds.do(send_email)  # send every 30 seconds for testing

print("⏳ Email scheduler started... (press CTRL+C to stop)")
while True:
    schedule.run_pending()
    time.sleep(1)
