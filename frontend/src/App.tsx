import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { DashboardPage } from "./pages/DashboardPage";
import { UploadPage } from "./pages/UploadPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { DocumentDetailPage } from "./pages/DocumentDetailPage";
import { ChunkExplorerPage } from "./pages/ChunkExplorerPage";
import { ReferenceDatasetPage } from "./pages/ReferenceDatasetPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { FeedbackPage } from "./pages/FeedbackPage";

export function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/documents/:id" element={<DocumentDetailPage />} />
        <Route path="/documents/:id/chunks" element={<ChunkExplorerPage />} />
        <Route path="/reference" element={<ReferenceDatasetPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/feedback" element={<FeedbackPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default App;
