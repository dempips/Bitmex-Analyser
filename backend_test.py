#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class TradeMetryxAPITester:
    def __init__(self, base_url="https://trademetryx.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}: {details}")

    def run_test(self, name: str, method: str, endpoint: str, expected_status: int, 
                 data: Optional[Dict] = None, params: Optional[Dict] = None) -> tuple[bool, Dict]:
        """Run a single API test"""
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, params=params, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)

            success = response.status_code == expected_status
            
            try:
                response_data = response.json() if response.text else {}
            except:
                response_data = {"raw_response": response.text}

            details = f"Status: {response.status_code} (expected {expected_status})"
            if not success:
                details += f", Response: {response.text[:200]}"
            
            self.log_result(name, success, details)
            return success, response_data

        except Exception as e:
            self.log_result(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_health_check(self) -> bool:
        """Test basic health endpoint"""
        success, _ = self.run_test("Health Check", "GET", "/health", 200)
        return success

    def test_register(self, email: str, password: str) -> bool:
        """Test user registration"""
        success, response = self.run_test(
            "User Registration",
            "POST",
            "/auth/register",
            200,
            data={"email": email, "password": password}
        )
        if success and 'token' in response:
            self.token = response['token']
            print(f"   Registered user: {email}")
            return True
        return False

    def test_login(self, email: str, password: str) -> bool:
        """Test user login"""
        success, response = self.run_test(
            "User Login",
            "POST",
            "/auth/login",
            200,
            data={"email": email, "password": password}
        )
        if success and 'token' in response:
            self.token = response['token']
            print(f"   Logged in user: {email}")
            return True
        return False

    def test_auth_me(self) -> bool:
        """Test authenticated user info"""
        success, response = self.run_test("Auth Me", "GET", "/auth/me", 200)
        if success and 'user' in response:
            print(f"   User info retrieved: {response['user'].get('email', 'N/A')}")
        return success

    def test_bitmex_symbols(self) -> bool:
        """Test BitMEX symbols endpoint"""
        success, response = self.run_test("BitMEX Symbols", "GET", "/bitmex/symbols", 200)
        if success and isinstance(response, list) and len(response) > 0:
            print(f"   Retrieved {len(response)} symbols")
            # Check if XBTUSD is in the list
            symbols = [s.get('symbol') for s in response if isinstance(s, dict)]
            if 'XBTUSD' in symbols:
                print(f"   ✓ XBTUSD found in symbols")
            else:
                print(f"   ⚠ XBTUSD not found in symbols")
        return success

    def test_analytics_snapshot(self, symbol: str = "XBTUSD") -> bool:
        """Test analytics snapshot endpoint"""
        success, response = self.run_test(
            "Analytics Snapshot",
            "GET",
            "/bitmex/analytics/snapshot",
            200,
            params={"symbol": symbol, "depth": 50, "bands_bps": "10,25,100"}
        )
        if success:
            required_fields = ['symbol', 'ts', 'best_bid', 'best_ask', 'mid', 'spread', 'bands']
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                print(f"   ⚠ Missing fields: {missing_fields}")
            else:
                print(f"   ✓ All required fields present")
                print(f"   Mid price: {response.get('mid', 'N/A')}")
        return success

    def test_analytics_flow(self, symbol: str = "XBTUSD") -> bool:
        """Test analytics flow endpoint"""
        success, response = self.run_test(
            "Analytics Flow",
            "GET",
            "/bitmex/analytics/flow",
            200,
            params={"symbol": symbol, "minutes": 5}
        )
        if success:
            required_fields = ['symbol', 'ts', 'minutes', 'buy_volume', 'sell_volume', 'aggressive_imbalance', 'cvd']
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                print(f"   ⚠ Missing fields: {missing_fields}")
            else:
                print(f"   ✓ All required fields present")
                print(f"   Buy volume: {response.get('buy_volume', 'N/A')}")
        return success

    def test_bitmex_candles(self, symbol: str = "XBTUSD") -> bool:
        """Test BitMEX candles endpoint"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)
        
        success, response = self.run_test(
            "BitMEX Candles",
            "GET",
            "/bitmex/candles",
            200,
            params={
                "symbol": symbol,
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        )
        if success and isinstance(response, list):
            print(f"   Retrieved {len(response)} candles")
            if len(response) > 0:
                candle = response[0]
                required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                missing_fields = [f for f in required_fields if f not in candle]
                if missing_fields:
                    print(f"   ⚠ Missing candle fields: {missing_fields}")
                else:
                    print(f"   ✓ Candle structure valid")
        return success

    def test_backtest_run(self, symbol: str = "XBTUSD") -> bool:
        """Test backtest run endpoint"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        strategy_payload = {
            "symbol": symbol,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "strategy": {
                "name": "Test Strategy",
                "symbol": symbol,
                "entry_conditions": [
                    {"metric": "close", "operator": ">", "value": 0}
                ],
                "exit_conditions": [
                    {"metric": "return_1", "operator": "<", "value": -0.01}
                ],
                "fee_bps": 7.5,
                "slippage_bps": 2.0
            },
            "initial_capital": 10000.0
        }
        
        success, response = self.run_test(
            "Backtest Run",
            "POST",
            "/backtests/run",
            200,
            data=strategy_payload
        )
        if success:
            required_fields = ['id', 'created_at', 'symbol', 'summary', 'equity_curve', 'trades']
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                print(f"   ⚠ Missing backtest fields: {missing_fields}")
            else:
                print(f"   ✓ Backtest completed successfully")
                summary = response.get('summary', {})
                print(f"   Total return: {summary.get('total_return_pct', 'N/A')}%")
                print(f"   Trades: {summary.get('trades', 'N/A')}")
        return success

    def test_protected_route_without_auth(self) -> bool:
        """Test that protected routes require authentication"""
        # Temporarily clear token
        original_token = self.token
        self.token = None
        
        success, _ = self.run_test("Protected Route (No Auth)", "GET", "/auth/me", 401)
        
        # Restore token
        self.token = original_token
        return success

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting TradeMetryx API Test Suite")
        print("=" * 50)
        
        # Generate unique test user
        timestamp = datetime.now().strftime("%H%M%S")
        test_email = f"test_user_{timestamp}@example.com"
        test_password = "TestPassword123!"
        
        # Test sequence
        tests = [
            ("Health Check", lambda: self.test_health_check()),
            ("User Registration", lambda: self.test_register(test_email, test_password)),
            ("Auth Me (After Register)", lambda: self.test_auth_me()),
            ("User Login", lambda: self.test_login(test_email, test_password)),
            ("Auth Me (After Login)", lambda: self.test_auth_me()),
            ("Protected Route Without Auth", lambda: self.test_protected_route_without_auth()),
            ("BitMEX Symbols", lambda: self.test_bitmex_symbols()),
            ("Analytics Snapshot", lambda: self.test_analytics_snapshot()),
            ("Analytics Flow", lambda: self.test_analytics_flow()),
            ("BitMEX Candles", lambda: self.test_bitmex_candles()),
            ("Backtest Run", lambda: self.test_backtest_run()),
        ]
        
        print(f"\nRunning {len(tests)} tests...")
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.log_result(test_name, False, f"Exception: {str(e)}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Total tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        
        # Print failed tests
        failed_tests = [r for r in self.test_results if not r['success']]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test['test']}: {test['details']}")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test runner"""
    tester = TradeMetryxAPITester()
    success = tester.run_all_tests()
    
    # Save detailed results
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_tests': tester.tests_run,
                'passed_tests': tester.tests_passed,
                'failed_tests': tester.tests_run - tester.tests_passed,
                'success_rate': (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': tester.test_results
        }, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())