#!/usr/bin/env python3
"""
Open QuantTrader Documentation in Browser
Opens the HTML documentation file in the default browser for easy PDF conversion
"""

import webbrowser
import os
import platform

def open_documentation():
    """Open the HTML documentation in the default browser"""
    
    # Get the current directory
    current_dir = os.getcwd()
    html_file = os.path.join(current_dir, 'QuantTrader_Documentation.html')
    
    # Check if the HTML file exists
    if not os.path.exists(html_file):
        print("❌ HTML documentation file not found!")
        print("💡 Make sure QuantTrader_Documentation.html exists in the current directory")
        return False
    
    # Convert file path to URL format
    if platform.system() == 'Windows':
        # Windows
        file_url = f"file:///{html_file.replace(os.sep, '/')}"
    else:
        # macOS and Linux
        file_url = f"file://{html_file}"
    
    try:
        # Open in default browser
        webbrowser.open(file_url)
        print("✅ Opening QuantTrader documentation in your default browser...")
        print("\n📖 To convert to PDF:")
        print("   1. Wait for the page to load completely")
        print("   2. Press Ctrl+P (Windows/Linux) or Cmd+P (Mac)")
        print("   3. Select 'Save as PDF' as the destination")
        print("   4. Click 'Save'")
        print("\n🎯 The documentation includes:")
        print("   • Complete system architecture")
        print("   • Detailed data flow diagrams")
        print("   • Step-by-step process breakdown")
        print("   • Technical specifications")
        print("   • Deployment guide")
        print("   • Troubleshooting tips")
        return True
        
    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print(f"💡 Try opening this file manually: {html_file}")
        return False

if __name__ == "__main__":
    print("🚀 QuantTrader Documentation Viewer")
    print("=" * 40)
    open_documentation() 