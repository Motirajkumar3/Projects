import { Link } from "react-router-dom";
import { Instagram, Facebook } from "lucide-react";

const Footer = () => (
  <footer className="mt-24 border-t border-border bg-secondary/40">
    <div className="container py-16 grid grid-cols-2 md:grid-cols-4 gap-10">
      <div className="col-span-2 md:col-span-1">
        <Link to="/" className="font-serif text-2xl">V<span className="text-accent">·</span>tailers</Link>
        <p className="mt-4 text-sm text-muted-foreground max-w-xs leading-relaxed">
          Custom tailoring crafted for the modern woman. Quiet luxury, made to measure.
        </p>
      </div>

      <div>
        <h4 className="text-xs uppercase tracking-luxury mb-4">Shop</h4>
        <ul className="space-y-2 text-sm text-muted-foreground">
          <li><Link to="/category/tops" className="hover:text-accent">Tops</Link></li>
          <li><Link to="/category/kurtis" className="hover:text-accent">Kurtis</Link></li>
          <li><Link to="/category/dresses" className="hover:text-accent">Dresses</Link></li>
          <li><Link to="/category/sets" className="hover:text-accent">Sets</Link></li>
        </ul>
      </div>

      <div>
        <h4 className="text-xs uppercase tracking-luxury mb-4">Atelier</h4>
        <ul className="space-y-2 text-sm text-muted-foreground">
          <li>Our Story</li>
          <li>Bespoke Service</li>
          <li>Size Guide</li>
          <li>Care</li>
        </ul>
      </div>

      <div>
        <h4 className="text-xs uppercase tracking-luxury mb-4">Newsletter</h4>
        <p className="text-sm text-muted-foreground mb-3">Receive new arrivals & private events.</p>
        <form className="flex border-b border-border focus-within:border-accent transition-colors">
          <input
            type="email"
            placeholder="Your email"
            className="flex-1 bg-transparent py-2 text-sm outline-none placeholder:text-muted-foreground"
          />
          <button type="submit" className="text-xs uppercase tracking-luxury text-accent hover:opacity-70">Join</button>
        </form>
        <div className="flex gap-4 mt-6 text-foreground/70">
          <a href="#" aria-label="Instagram" className="hover:text-accent"><Instagram className="h-4 w-4" /></a>
          <a href="#" aria-label="Facebook" className="hover:text-accent"><Facebook className="h-4 w-4" /></a>
        </div>
      </div>
    </div>
    <div className="border-t border-border py-6">
      <div className="container flex flex-col md:flex-row justify-between items-center gap-2 text-xs text-muted-foreground">
        <p>© {new Date().getFullYear()} V·tailers. All rights reserved.</p>
        <p>Crafted in India</p>
      </div>
    </div>
  </footer>
);

export default Footer;
