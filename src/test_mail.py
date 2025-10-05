import smtplib
from email.mime.text import MIMEText
import schedule
import time

# -------------------
# Email configuration
# -------------------
SENDER_EMAIL = "udaygupta.ph@gmail.com"     # Replace with your email
APP_PASSWORD = "xqga frbv ouol bnck"   # Replace with generated app password
RECEIVER_EMAIL = "guptauday49@gmail.com"  # Replace with recipient email

def send_email():
    """Send a test email."""
    msg = MIMEText("Hello! This is a test automated email.")
    msg["Subject"] = "Test Email"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("✅ Email sent successfully")
    except Exception as e:
        print("❌ Error sending email:", e)

# -------------------
# Schedule job
# -------------------
schedule.every(1).minutes.do(send_email)  # send every 1 minute for testing

print("⏳ Email scheduler started... (press CTRL+C to stop)")
while True:
    schedule.run_pending()
    time.sleep(1)
