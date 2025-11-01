import { Github, Linkedin, Mail } from "lucide-react";
import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="border-t border-border/50 bg-card/50 backdrop-blur-sm">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Link to="/" className="flex items-center">
                <img
                  src={`${import.meta.env.BASE_URL}WUhop.png`}
                  alt="UHOP"
                  className="h-10 w-auto object-contain"
                />
              </Link>
            </div>
            <p className="text-sm text-muted-foreground">
              AI-Generated Kernel Intelligence — Optimize once, run anywhere.
            </p>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Resources</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a
                  href="/docs"
                  className="hover:text-primary transition-colors"
                >
                  Documentation
                </a>
              </li>
              <li>
                <a
                  href="/demo"
                  className="hover:text-primary transition-colors"
                >
                  Demo
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/sevenloops/uhop"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-primary transition-colors"
                >
                  GitHub
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Community</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a href="#" className="hover:text-primary transition-colors">
                  Discussions
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition-colors">
                  Contributing
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition-colors">
                  Code of Conduct
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4">Connect</h3>
            <div className="flex space-x-4">
              <a
                href="https://github.com/sevenloops"
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                <Linkedin className="h-5 w-5" />
              </a>
              <a
                href="mailto:contact@uhop.dev"
                className="text-muted-foreground hover:text-primary transition-colors"
              >
                <Mail className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-border/50 text-center text-sm text-muted-foreground">
          <p>
            &copy; {new Date().getFullYear()} UHop. Built by Sevenloops. Open
            source under MIT License.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
