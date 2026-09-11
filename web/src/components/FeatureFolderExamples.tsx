import './FeatureFolderExamples.css';

export default function FeatureFolderExamples() {
  return (
    <details className="feature-folder-examples">
      <summary>Example folder structures</summary>
      <div className="feature-folder-examples-body">
        <p>These folder names and paths are examples. Choose the corresponding folder on your server.</p>
        <div className="feature-folder-example-grid">
          <section className="feature-folder-example">
            <h3>TRIDENT job root</h3>
            <p>The parent folder containing patch settings and model outputs.</p>
            <pre tabIndex={0} aria-label="Example TRIDENT job folder structure"><code><strong>trident_output/  ← select this folder</strong>{`
├── contours_geojson/
└── 20x_256px_0px_overlap/
    ├── patches/
    │   ├── slide_001_patches.h5
    │   └── slide_002_patches.h5
    └── features_<encoder>/
        ├── slide_001.h5
        └── slide_002.h5`}</code></pre>
            <p className="feature-folder-example-path"><span>Example path</span><code>/path/to/trident_output</code></p>
          </section>
          <section className="feature-folder-example">
            <h3>Feature directory</h3>
            <p>The folder containing one feature file per slide, for one model.</p>
            <pre tabIndex={0} aria-label="Example feature directory structure"><code><strong>features_&lt;encoder&gt;/  ← select this folder</strong>{`
├── slide_001.h5
└── slide_002.h5`}</code></pre>
            <p className="feature-folder-example-path"><span>Example path</span><code>/path/to/trident_output/20x_256px_0px_overlap/features_&lt;encoder&gt;</code></p>
            <p>A simple folder such as <code>my_features/</code> containing the same per-slide files also works.</p>
          </section>
        </div>
        <p>You can also select the patch settings folder: <code>trident_output/20x_256px_0px_overlap/</code> means 20× magnification, 256-pixel patches, and no overlap.</p>
        <p><code>&lt;encoder&gt;</code> stands for the name of the model used to extract the features.</p>
        <p>If a job contains several models or patch settings, select the specific <code>features_&lt;encoder&gt;/</code> folder you want to attach.</p>
        <details className="feature-file-example">
          <summary>What goes inside each .h5 file?</summary>
          <p><code>slide_001.h5</code> matches <strong>Slide_ID slide_001</strong> in your dataset. Each file stores patch features for that slide.</p>
          <dl>
            <div><dt><code>features [N, D]</code></dt><dd>A floating-point array with one row per patch. N is the patch count; D is the number of feature values per patch.</dd></div>
            <div><dt><code>coords [N, 2]</code></dt><dd>One integer x/y coordinate pair per feature row.</dd></div>
          </dl>
          <p>Coordinates can be embedded in <code>slide_001.h5</code> or stored separately in <code>patches/slide_001_patches.h5</code>. If they are not found automatically, select the <code>patches/</code> folder under <strong>Advanced Options → Coordinate directory</strong>.</p>
          <p>Patch counts may differ between slides; the feature dimension must stay consistent. These examples describe patch features used for MIL.</p>
        </details>
      </div>
    </details>
  );
}
