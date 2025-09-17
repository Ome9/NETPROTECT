#!/usr/bin/env node
/**
 * Test script to verify NetProtect Backend API is working
 */

const http = require('http');

console.log('🔍 Testing NetProtect Backend API...');

// Test basic API endpoint
const testAPI = () => {
  const options = {
    hostname: 'localhost',
    port: 3001,
    path: '/api',
    method: 'GET'
  };

  const req = http.request(options, (res) => {
    let data = '';
    
    res.on('data', (chunk) => {
      data += chunk;
    });
    
    res.on('end', () => {
      if (res.statusCode === 200) {
        console.log('✅ API Root Endpoint Working');
        console.log('📊 Response:', JSON.parse(data));
        testNetworkData();
      } else {
        console.log(`❌ API Root Failed: ${res.statusCode}`);
        console.log('Response:', data);
      }
    });
  });

  req.on('error', (err) => {
    console.log('❌ Backend Server Not Running');
    console.log('💡 Start the backend server with: npm run dev');
    console.log('Error:', err.message);
  });

  req.end();
};

// Test network data endpoint
const testNetworkData = () => {
  const options = {
    hostname: 'localhost',
    port: 3001,
    path: '/api/network/current',
    method: 'GET'
  };

  const req = http.request(options, (res) => {
    let data = '';
    
    res.on('data', (chunk) => {
      data += chunk;
    });
    
    res.on('end', () => {
      if (res.statusCode === 200) {
        console.log('✅ Network Data Endpoint Working');
        const response = JSON.parse(data);
        console.log('📈 Raw Response:', JSON.stringify(response, null, 2));
        console.log('📈 Network Stats:');
        
        // Check the actual structure of the response
        if (response.connections !== undefined) {
          console.log(`  - Connections: ${response.connections || 0}`);
          console.log(`  - CPU Usage: ${response.cpuUsage || 0}%`);
          console.log(`  - Memory Usage: ${response.memoryUsage || 0}%`);
          console.log(`  - Network Load: ${response.networkLoad || 0}%`);
          console.log(`  - Threats Blocked: ${response.threatsBlocked || 0}`);
        } else if (response.data && response.data.connections !== undefined) {
          console.log(`  - Connections: ${response.data.connections}`);
          console.log(`  - Bytes In: ${response.data.bytesIn}`);
          console.log(`  - Bytes Out: ${response.data.bytesOut}`);
          console.log(`  - TCP Connections: ${response.data.protocols.TCP}`);
        } else {
          console.log('  - Available keys:', Object.keys(response));
        }
        console.log('🎉 Backend API is working properly!');
      } else {
        console.log(`❌ Network Data Failed: ${res.statusCode}`);
        console.log('Response:', data);
      }
    });
  });

  req.on('error', (err) => {
    console.log('❌ Network Data Endpoint Error:', err.message);
  });

  req.end();
};

// Start testing
testAPI();