import { useState } from "react";
import "./App.css";
import LiveKitModal from "./components/LiveKitModal";

function App() {
  const [showSupport, setShowSupport] = useState(false);

  return (
    <div className="app">
      <header className="header">
        <div className="logo">Live Agent Support</div>
      </header>

      <main>
        <section className="features">
          <button
            className="support-button"
            onClick={() => setShowSupport(true)}
          >
            Talk to an Agent!
          </button>
        </section>
      </main>

      {showSupport && <LiveKitModal setShowSupport={setShowSupport} />}
    </div>
  );
}

export default App;
