import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(subject, body, to_email):
    # SMTP server settings
    smtp_server = 'smtp.yourserver.com'
    smtp_port = 587  # or 465 for SSL/TLS
    smtp_username = 'your_username'
    smtp_password = 'your_password'

    # Create message
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Connect to SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # For SSL/TLS connection
    server.login(smtp_username, smtp_password)

    # Send email
    server.sendmail(smtp_username, to_email, msg.as_string())

    # Close connection
    server.quit()

# Example usage
send_email('Theft Detected', 'A theft has been detected at your premises.', 'recipient@example.com')
