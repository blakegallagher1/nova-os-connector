/**
 * Jest test setup file
 * Runs before each test file
 */

// Set test environment variables
process.env.GITHUB_PERSONAL_ACCESS_TOKEN = 'test_token_for_testing_only';
process.env.PORT = '8001'; // Use different port for tests
process.env.BUILD_COMMAND = 'npm run build';
process.env.MAX_CONCURRENT_REQUESTS = '3';

// Suppress console output during tests (optional - comment out for debugging)
// global.console = {
//   ...console,
//   log: jest.fn(),
//   error: jest.fn(),
//   warn: jest.fn(),
// };

export {};
