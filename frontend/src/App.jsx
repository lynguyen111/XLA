import { useState, useCallback } from "react";
import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import UploadZone from "./components/UploadZone";
import PredictionResult from "./components/PredictionResult";

export default function App() {
  const [model, setModel] = useState("resnet18");
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = useCallback((file, previewUrl) => {
    setImageFile(file);
    setImage(previewUrl);
    setPredictions(null);
    setError(null);
  }, []);

  const handlePredict = useCallback(async () => {
    if (!imageFile) return;

    setLoading(true);
    setError(null);
    setPredictions(null);

    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("model", model);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Lỗi không xác định");
      setPredictions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [imageFile, model]);

  const handleReset = useCallback(() => {
    setImage(null);
    setImageFile(null);
    setPredictions(null);
    setError(null);
  }, []);

  return (
    <div className="app">
      <Header />
      <main className="main">
        <div className="container">
          <ModelSelector selected={model} onChange={setModel} />

          <div className="workspace">
            <div className="upload-section">
              <UploadZone
                onImageSelect={handleImageSelect}
                preview={image}
                onReset={handleReset}
              />
              {image && !loading && (
                <button
                  className="btn-predict"
                  onClick={handlePredict}
                  disabled={loading}
                >
                  Nhận Diện
                </button>
              )}
            </div>

            <div className="result-section">
              {loading && (
                <div className="loading-box">
                  <div className="spinner" />
                  <p>Đang phân tích ảnh...</p>
                </div>
              )}
              {error && (
                <div className="error-box">
                  <span className="error-icon">⚠️</span>
                  <p>{error}</p>
                </div>
              )}
              {predictions && !loading && (
                <PredictionResult data={predictions} />
              )}
              {!image && !loading && !predictions && (
                <div className="placeholder">
                  <div className="placeholder-icon">🔍</div>
                  <p>Tải ảnh côn trùng lên để bắt đầu nhận diện</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
