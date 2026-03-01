
import pandas as pd
from datetime import datetime
import os

class AttendanceManager:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        if not os.path.exists(excel_path):
            df = pd.DataFrame(
                columns=['Student_ID', 'Name', 'Date', 'Time', 'Status', 'Count']
            )
            df.to_excel(excel_path, index=False)

    def mark(self, student_id, name):
        today = datetime.now().strftime('%Y-%m-%d')
        now = datetime.now().strftime('%H:%M:%S')
        
        try:
            df = pd.read_excel(self.excel_path)
            
            # Ensure Count column exists for backward compatibility
            if 'Count' not in df.columns:
                df['Count'] = 1
        except Exception:
             # Handle empty or corrupted file by recreating
             df = pd.DataFrame(columns=['Student_ID', 'Name', 'Date', 'Time', 'Status', 'Count'])

        # Check if attendance already marked for this student today
        mask = (df['Student_ID'] == student_id) & (df['Date'] == today)
        
        if mask.any():
            # Get the index of the existing row
            idx = df.index[mask][0]
            
            # Increment count
            current_count = df.at[idx, 'Count']
            df.at[idx, 'Count'] = current_count + 1
            
            # Update time to latest check-in
            df.at[idx, 'Time'] = now
            
            print(f"[INFO] Attendance already exists. Incrementing count to {df.at[idx, 'Count']}.")
        else:
            # Create new row
            new_row = {
                'Student_ID': student_id,
                'Name': name,
                'Date': today,
                'Time': now,
                'Status': 'Present',
                'Count': 1
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"[INFO] New attendance marked for {name}.")

        df.to_excel(self.excel_path, index=False)
