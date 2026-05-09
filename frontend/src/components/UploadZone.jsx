import { useRef, useState, useCallback } from "react";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function UploadZone({ onImageSelect, preview, onReset }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const processFile = useCallback(
    (file) => {
      if (!file || !ACCEPTED.includes(file.type)) {
        alert("Chỉ hỗ trợ ảnh JPG, PNG, WEBP, BMP");
        return;
      }
      const url = URL.createObjectURL(file);
      onImageSelect(file, url);
    },
    [onImageSelect]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      processFile(file);
    },
    [processFile]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => setDragging(false);

  const handleFileChange = (e) => {
    processFile(e.target.files[0]);
    e.target.value = "";
  };

  if (preview) {
    return (
      <div className="preview-wrapper">
        <img src={preview} alt="Preview" className="preview-img" />
        <button className="btn-reset" onClick={onReset} title="Xóa ảnh">
          ✕
        </button>
      </div>
    );
  }

  return (
    <div
      className={`upload-zone ${dragging ? "dragging" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      <div className="upload-icon">📷</div>
      <p className="upload-text">Kéo thả ảnh vào đây</p>
      <p className="upload-sub">hoặc nhấn để chọn file</p>
      <p className="upload-hint">Hỗ trợ: JPG, PNG, WEBP, BMP</p>
    </div>
  );
}
