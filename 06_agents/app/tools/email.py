EMAIL_TOOL_DESCRIPTION = """
Tool Name: send_email
Description: Useful for sending an email to a user or client. 
Action Input must be formatted exactly as: email_address|subject|body (separated by pipes).
"""

def execute_send_email(input_str: str) -> str:
    """Mock function to send an email."""
    try:
        email, subject, body = input_str.split("|")
        # In a real app, you would use SMTP, SendGrid, AWS SES, etc. here.
        print(f"\n[MOCK EMAIL SENT] To: {email} | Subject: {subject} | Body: {body}\n")
        return f"Email successfully sent to {email}."
    except ValueError:
        return "Error: Input must be formatted as email|subject|body"
    