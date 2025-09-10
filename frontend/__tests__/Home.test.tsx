import { render, screen } from '@testing-library/react';
import Home from '../src/app/page';

describe('Home', () => {
  it('renders heading', () => {
    render(<Home />);
    const heading = screen.getByRole('heading', {
      name: /楽しみながら.*非認知能力.*を育む/i,
    });
    expect(heading).toBeInTheDocument();
  });
});
