from pathlib import Path

# Set up test data paths
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"
TEST_OUTPUT_DIR = TEST_DIR / "test_output"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)
