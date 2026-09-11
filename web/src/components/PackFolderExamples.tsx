import './FeatureFolderExamples.css';

export default function PackFolderExamples() {
  return (
    <details className="feature-folder-examples">
      <summary>What does a pack folder look like?</summary>
      <div className="feature-folder-examples-body">
        <p>A pack gathers the per-slide features into shared arrays with an index. It is a separate, indexed copy of one feature version that can be included in a frozen feature bundle. Choose the folder containing the files below, for the same encoder and patch configuration.</p>
        <div className="feature-folder-example-grid">
          <section className="feature-folder-example">
            <h4>HistoPilot pack</h4>
            <pre tabIndex={0} aria-label="Example HistoPilot pack folder structure"><code><strong>my_feature_pack/  ← select this folder</strong>{`
├── features.bin
├── coords.bin
├── index.parquet
├── meta.json
├── manifest.json
└── checksums.json`}</code></pre>
            <p className="feature-folder-example-path"><span>Example path</span><code>/path/to/packs/my_feature_pack</code></p>
          </section>
          <section className="feature-folder-example">
            <h4>Existing OceanPath v1 pack</h4>
            <pre tabIndex={0} aria-label="Example OceanPath pack folder structure"><code><strong>blca/  ← select this folder</strong>{`
├── features.bin
├── coords.bin
├── index.parquet
└── meta.json`}</code></pre>
            <p className="feature-folder-example-path"><span>Example path</span><code>/path/to/mmap/blca</code></p>
          </section>
        </div>
        <dl>
          <div><dt><code>features.bin</code></dt><dd>Feature values for every patch, laid out using the dtype and dimensions recorded in the metadata.</dd></div>
          <div><dt><code>coords.bin</code></dt><dd>The matching x/y patch coordinates, in the same row order.</dd></div>
          <div><dt><code>index.parquet</code></dt><dd>Slide IDs, patch counts and offsets that locate each slide’s rows.</dd></div>
          <div><dt><code>meta.json</code></dt><dd>Pack format, precision, dimensions and overall counts.</dd></div>
          <div><dt><code>manifest.json</code></dt><dd>Additional HistoPilot checksums and provenance. Legacy OceanPath packs can be verified without this file.</dd></div>
          <div><dt><code>checksums.json</code></dt><dd>Content hashes used to detect changes to a HistoPilot pack’s files.</dd></div>
        </dl>
        <p>Select <code>my_feature_pack/</code> or <code>blca/</code> directly. Keep the original feature folder available: HistoPilot compares slide IDs, patch counts, precision and payload sizes, then verifies all feature values and coordinates before allowing inclusion in a bundle.</p>
      </div>
    </details>
  );
}
