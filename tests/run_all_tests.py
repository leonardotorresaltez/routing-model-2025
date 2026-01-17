"""
Master test runner - runs all tests in the test suite.

Execute this file to run all tests in order:
1. Graph Pointer integration tests
2. Environment tests
3. Final integration tests
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all():
    """Run all tests in sequence."""
    print("\n" + "=" * 70)
    print("RUNNING COMPLETE TEST SUITE")
    print("=" * 70 + "\n")
    
    all_passed = True
    
    print("[1/3] Running Graph Pointer Integration Tests...")
    print("-" * 70)
    try:
        from tests.test_graph_pointer_integration import run_all_tests as run_graph_tests
        run_graph_tests()
        print("[PASS] Graph Pointer tests completed\n")
    except Exception as e:
        print(f"[FAIL] Graph Pointer tests: {e}\n")
        all_passed = False
    
    print("[2/3] Running Environment Tests...")
    print("-" * 70)
    try:
        from tests.test_environment import (test_environment_basics, test_feasibility_masking,
                                            test_episode_completion, test_baseline_policies)
        env, customers, trucks, depots = test_environment_basics()
        test_feasibility_masking(env)
        test_episode_completion(env)
        test_baseline_policies(customers, trucks, depots)
        print("[PASS] Environment tests completed\n")
    except Exception as e:
        print(f"[FAIL] Environment tests: {e}\n")
        all_passed = False
    
    print("[3/3] Running Final Integration Tests...")
    print("-" * 70)
    try:
        from tests.final_integration_test import test_complete_workflow
        if test_complete_workflow():
            print("[PASS] Final integration tests completed\n")
        else:
            print("[FAIL] Final integration tests failed\n")
            all_passed = False
    except Exception as e:
        print(f"[FAIL] Final integration tests: {e}\n")
        all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("SUCCESS - ALL TESTS PASSED")
        print("=" * 70)
        print("\nYour fleet routing system is fully operational.")
        print("Ready to train with: python train_ppo.py")
        return 0
    else:
        print("FAILURE - SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_all()
    sys.exit(exit_code)
