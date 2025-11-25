import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Configuration
# ----------------------------
SENDER_EMAIL = "uday.gupta@renataiot.com"
RECEIVER_EMAIL = "uday.gupta@renataiot.com"
CC_EMAILS = ["someone@example.com"]  # Add CC emails here
APP_PASSWORD = os.getenv("APP_PASSWORD")

SUBJECT = f"RenataAI Coupon Detection Report - {datetime.now().strftime('%d-%b-%Y')}"
#CSV_FILE_PATH = "src/results.csv"
CSV_FILE_PATH = f"detection_log_{datetime.now().strftime('%d_%b_%Y')}.csv"
LOGO_PATH = "src/logo_left.png"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587  # TLS

# HTML body with inline logo
BODY_HTML = f"""
<html>
  <body>
    <p>Date: {datetime.now().strftime('%A, %B %d, %Y')}</p>
    <p>Sir,</p>
    <p>PFA today's coupon detection report.</p>
    <br>
    <p>Thanks & Regards,<br>
       Uday Gupta<br>
       <br>
       <img src="cid:companylogo" 
            width="250" 
            style="display:block; margin-top:20px; margin-bottom:10px;">
    </p>
  </body>
</html>
"""

# ----------------------------
# Function to send email
# ----------------------------
def send_csv_email(sender_email, receiver_email, cc_emails, subject, body_html, csv_file_path, logo_path, smtp_server, smtp_port, password):
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Cc'] = ', '.join(cc_emails)
    msg['Subject'] = subject

    # Attach HTML body
    msg.attach(MIMEText(body_html, 'html'))

    # Attach CSV file
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(csv_file_path)}"')
            msg.attach(part)
    else:
        print(f"Warning: CSV file not found at {csv_file_path}")

    # Attach logo image inline
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as img:
            mime_img = MIMEImage(img.read())
            mime_img.add_header('Content-ID', '<companylogo>')
            mime_img.add_header('Content-Disposition', 'inline', filename=os.path.basename(logo_path))
            msg.attach(mime_img)
    else:
        print(f"Warning: Logo file not found at {logo_path}")

    # Combine To + CC for sending
    all_recipients = [receiver_email] + cc_emails

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, all_recipients, msg.as_string())

    print("Email sent successfully!")

# ----------------------------
# Call function
# ----------------------------
send_csv_email(
    sender_email=SENDER_EMAIL,
    receiver_email=RECEIVER_EMAIL,
    cc_emails=CC_EMAILS,
    subject=SUBJECT,
    body_html=BODY_HTML,
    csv_file_path=CSV_FILE_PATH,
    logo_path=LOGO_PATH,
    smtp_server=SMTP_SERVER,
    smtp_port=SMTP_PORT,
    password=APP_PASSWORD
)
