import { createBrowserRouter, Link, RouterProvider } from "react-router";
import { Layout } from "./components/Layout";
import { AskPage } from "./pages/AskPage";
import { ComparePage } from "./pages/ComparePage";
import { HomePage } from "./pages/HomePage";
import { ScoresPage } from "./pages/ScoresPage";

function NotFound() {
  return (
    <div className="py-16 text-center">
      <h1 className="text-xl font-semibold">Page not found</h1>
      <Link to="/" className="mt-2 inline-block text-sm text-accent underline">Back to home</Link>
    </div>
  );
}

export const routes = [
  {
    element: <Layout />,
    children: [
      { index: true, element: <HomePage /> },
      { path: "ask", element: <AskPage /> },
      { path: "compare", element: <ComparePage /> },
      { path: "scores", element: <ScoresPage /> },
      { path: "*", element: <NotFound /> },
    ],
  },
];

export function App() {
  return <RouterProvider router={createBrowserRouter(routes)} />;
}
