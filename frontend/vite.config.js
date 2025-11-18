import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import yaml from 'js-yaml'

// ES 모듈에서 __dirname 대체
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// config.yaml에서 API 설정 읽기
function loadConfig() {
  try {
    const configPath = path.resolve(__dirname, '../config/config.yaml')
    const configFile = fs.readFileSync(configPath, 'utf8')
    const config = yaml.load(configFile)
    
    const apiHost = config?.api?.host || '0.0.0.0'
    const apiPort = config?.api?.port || 8000
    
    // 0.0.0.0은 localhost로 변환
    const apiBaseUrl = apiHost === '0.0.0.0' 
      ? `http://localhost:${apiPort}`
      : `http://${apiHost}:${apiPort}`
    
    console.log(`📡 API 서버 설정: ${apiBaseUrl} (config.yaml에서 읽음)`)
    
    return {
      apiBaseUrl,
      apiHost,
      apiPort
    }
  } catch (error) {
    console.warn('⚠️ config.yaml을 읽을 수 없습니다. 기본값을 사용합니다.', error)
    return {
      apiBaseUrl: 'http://localhost:8000',
      apiHost: 'localhost',
      apiPort: 8000
    }
  }
}

const { apiBaseUrl } = loadConfig()

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // 네트워크 접속 허용 (모든 네트워크 인터페이스)
    port: 5173,
    strictPort: false, // 포트가 사용 중이면 다른 포트 사용
    proxy: {
      '/api': {
        target: apiBaseUrl,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, res) => {
            console.log('⚠️ 프록시 오류:', err.message);
            console.log('💡 API 서버가 실행 중인지 확인하세요.');
            console.log(`   실행 명령: python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${loadConfig().apiPort} --reload`);
          });
        }
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})

