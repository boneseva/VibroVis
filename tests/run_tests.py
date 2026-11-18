#!/usr/bin/env python
"""
Simple test runner script for VibroVis UI tests.
This script provides an easy way to run tests with common options.
"""
import sys
import subprocess
import argparse


def run_tests(args):
    """Run pytest with the given arguments."""
    pytest_args = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")
    else:
        pytest_args.append("-v")
    
    # Add specific test files or markers
    if args.file:
        pytest_args.append(args.file)
    elif args.mark:
        pytest_args.extend(["-m", args.mark])
    elif args.basic_only:
        pytest_args.append("tests/test_ui_basic.py")
    elif args.interactions_only:
        pytest_args.append("tests/test_ui_interactions.py")
    else:
        pytest_args.append("tests/")
    
    # Add browser option
    if args.browser:
        pytest_args.extend(["--browser", args.browser])
    
    # Add other options
    if args.headed:
        pytest_args.append("--headed")
    
    if args.slow:
        pytest_args.append("-m")
        pytest_args.append("slow")
    
    if args.no_slow:
        pytest_args.extend(["-m", "not slow"])
    
    # Run pytest
    print(f"Running: {' '.join(pytest_args)}")
    print("-" * 60)
    
    result = subprocess.run([sys.executable, "-m", "pytest"] + pytest_args[1:])
    return result.returncode


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Run VibroVis UI tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --basic-only       # Run only basic tests
  python run_tests.py --browser firefox  # Run in Firefox
  python run_tests.py --no-slow          # Skip slow tests
  python run_tests.py -m integration     # Run integration tests only
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (default)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet output"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Run specific test file"
    )
    
    parser.add_argument(
        "-m", "--mark",
        type=str,
        help="Run tests with specific marker (e.g., 'ui', 'integration')"
    )
    
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Run only basic UI tests"
    )
    
    parser.add_argument(
        "--interactions-only",
        action="store_true",
        help="Run only interaction tests"
    )
    
    parser.add_argument(
        "-b", "--browser",
        type=str,
        choices=["chromium", "firefox", "webkit"],
        help="Browser to use (default: chromium)"
    )
    
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode (visible)"
    )
    
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run only slow tests"
    )
    
    parser.add_argument(
        "--no-slow",
        action="store_true",
        help="Skip slow tests"
    )
    
    args = parser.parse_args()
    
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
