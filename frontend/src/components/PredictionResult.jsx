const MODEL_LABELS = {
  resnet18: "ResNet-18",
  densenet121: "DenseNet-121",
};

const CONFIDENCE_COLOR = (conf) => {
  if (conf >= 70) return "#22c55e";
  if (conf >= 40) return "#f59e0b";
  return "#ef4444";
};

export default function PredictionResult({ data }) {
  const { predictions, model, device } = data;
  const top = predictions[0];

  return (
    <div className="result-card">
      <div className="result-header">
        <div className="result-top-label">Kết quả nhận diện</div>
        <div className="result-meta">
          {MODEL_LABELS[model] || model} · {device.toUpperCase()}
        </div>
      </div>

      <div className="result-winner">
        <span className="winner-emoji">{top.emoji}</span>
        <div>
          <div className="winner-name">{top.display_name}</div>
          <div className="winner-conf">
            Độ tin cậy:{" "}
            <strong style={{ color: CONFIDENCE_COLOR(top.confidence) }}>
              {top.confidence}%
            </strong>
          </div>
        </div>
      </div>

      <div className="result-list-title">Top 5 dự đoán</div>
      <div className="result-list">
        {predictions.map((p) => (
          <div key={p.class_id} className={`result-item ${p.rank === 1 ? "top" : ""}`}>
            <div className="item-header">
              <span className="item-emoji">{p.emoji}</span>
              <span className="item-name">{p.display_name}</span>
              <span
                className="item-conf"
                style={{ color: CONFIDENCE_COLOR(p.confidence) }}
              >
                {p.confidence}%
              </span>
            </div>
            <div className="item-bar-bg">
              <div
                className="item-bar-fill"
                style={{
                  width: `${p.confidence}%`,
                  backgroundColor: CONFIDENCE_COLOR(p.confidence),
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
