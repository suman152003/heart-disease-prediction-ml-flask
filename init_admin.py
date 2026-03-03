"""
Initialize Admin User
Run this script to create an admin user in the database.
Default admin: username='admin', password='admin@123'
"""
from app import app, db, Admin

def init_admin():
    with app.app_context():
        # Default admin credentials
        default_username = "admin"
        default_password = "admin@123"
        
        # Check if default admin already exists
        existing_admin = Admin.query.filter_by(username=default_username).first()
        if existing_admin:
            print(f"Default admin user '{default_username}' already exists!")
            print("Using existing admin credentials:")
            print(f"  Username: {default_username}")
            print(f"  Password: {default_password}")
            return
        
        # Check if any admin exists
        any_admin = Admin.query.first()
        if any_admin:
            print("Admin user(s) already exist!")
            print("Existing admin usernames:")
            for admin in Admin.query.all():
                print(f"  - {admin.username}")
            response = input(f"\nDo you want to create the default admin '{default_username}'? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Create default admin
        print(f"Creating default admin user...")
        admin = Admin(username=default_username, password=default_password)
        db.session.add(admin)
        db.session.commit()
        print(f"\n✓ Default admin user created successfully!")
        print(f"  Username: {default_username}")
        print(f"  Password: {default_password}")
        print("\nYou can now login to the admin dashboard using these credentials.")

if __name__ == "__main__":
    init_admin()

