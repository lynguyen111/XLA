const MODELS = [
  {
    id: "resnet18",
    name: "ResNet-18",
    description: "Nhanh, nhẹ",
  },
  {
    id: "densenet121",
    name: "DenseNet-121",
    description: "Chính xác cao",
  },
];

export default function ModelSelector({ selected, onChange }) {
  return (
    <div className="model-selector">
      <p className="model-label">Chọn mô hình:</p>
      <div className="model-tabs">
        {MODELS.map((m) => (
          <button
            key={m.id}
            className={`model-tab ${selected === m.id ? "active" : ""}`}
            onClick={() => onChange(m.id)}
          >
            <span className="tab-name">{m.name}</span>
            <span className="tab-desc">{m.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
