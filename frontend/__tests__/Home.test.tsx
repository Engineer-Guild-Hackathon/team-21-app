import { render, screen } from "@testing-library/react";
import Home from "../src/app/page";

describe("Home", () => {
  it("renders heading", () => {
    render(<Home />);
    const heading = screen.getByRole("heading", {
      name: /非認知能力学習プラットフォーム/i,
    });
    expect(heading).toBeInTheDocument();
  });
});
