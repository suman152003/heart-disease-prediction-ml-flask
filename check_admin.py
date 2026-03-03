"""
Check existing admin accounts or create a new one
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Admin

def check_or_create_admin():
    with app.app_context():
        # Check existing admins
        admins = Admin.query.all()
        
        if admins:
            print("\n" + "="*50)
            print("EXISTING ADMIN ACCOUNTS:")
            print("="*50)
            for admin in admins:
                print(f"Username: {admin.username}")
            print("="*50)
            print("\nIf you forgot the password, you'll need to reset it manually in the database.")
        else:
            print("\n" + "="*50)
            print("NO ADMIN ACCOUNTS FOUND")
            print("="*50)
            print("\nYou need to create an admin account.")
            print("Run: python init_admin.py")
            print("="*50)

if __name__ == "__main__":
    check_or_create_admin()

