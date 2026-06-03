from app import has_data
import sys
if not has_data():
    sys.exit(1)
print('ok')
