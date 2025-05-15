'use client';

import { useState } from 'react';

export default function Home() {
  const [url, setUrl] = useState('');
  const [label, setLabel] = useState('');
  const [confidence, setConfidence] = useState<number | null>(null);
  const [screenshot, setScreenshot] = useState('');
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8080/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      const data = await res.json();
      setLabel(data.label);
      setConfidence(data.confidence);
      setScreenshot(`data:image/png;base64,${data.screenshot_base64}`);
    } catch (err) {
      alert('Lỗi: Không thể phân tích website');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-4">Phân tích Website Defacement</h1>

      <input
        type="text"
        placeholder="Nhập URL (vd: example.com)"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        className="border p-2 w-full mb-4 rounded"
      />

      <button
        onClick={handlePredict}
        disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        {loading ? 'Đang phân tích...' : 'Dự đoán'}
      </button>

      {label && (
        <div className="mt-6">
          <p className="text-xl">
            <strong>Kết quả:</strong>{' '}
            <span className={label === 'defaced' ? 'text-red-500' : 'text-green-600'}>
              {label.toUpperCase()}
            </span>
          </p>
          <p className="text-sm text-gray-600">Độ tin cậy: {confidence?.toFixed(2)}</p>

          {screenshot && (
            <div className="mt-4">
              <strong>Ảnh chụp màn hình:</strong>
              <img src={screenshot} alt="Ảnh chụp màn hình" className="rounded shadow" />
            </div>
          )}
        </div>
      )}
    </main>
  );
}
