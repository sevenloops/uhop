import { Card } from "@/components/ui/card";
import { Github, Linkedin, Mail, Target, Zap, Users } from "lucide-react";
import { Button } from "@/components/ui/button";

const About = () => {
  return (
    <div className="min-h-screen pt-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="max-w-4xl mx-auto space-y-12">
          {/* Mission Section */}
          <section className="text-center space-y-4">
            <h1 className="text-4xl sm:text-5xl font-bold gradient-text">
              Our Mission
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Democratizing hardware performance through AI-driven optimization
            </p>
          </section>

          <Card className="p-8 bg-card/50 backdrop-blur-sm border-primary/20">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  icon: Target,
                  title: "Openness",
                  desc: "Open source foundation enabling community-driven innovation"
                },
                {
                  icon: Zap,
                  title: "Performance",
                  desc: "Relentless focus on achieving optimal hardware utilization"
                },
                {
                  icon: Users,
                  title: "Universality",
                  desc: "One protocol that works seamlessly across all hardware"
                }
              ].map((value, idx) => (
                <div key={idx} className="text-center space-y-3">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10">
                    <value.icon className="h-8 w-8 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold">{value.title}</h3>
                  <p className="text-sm text-muted-foreground">{value.desc}</p>
                </div>
              ))}
            </div>
          </Card>

          {/* Vision Section */}
          <section className="space-y-6">
            <h2 className="text-3xl font-bold text-center">The Vision</h2>
            <Card className="p-6 bg-card/50">
              <p className="text-muted-foreground leading-relaxed mb-4">
                UHop represents the foundation of a future where developers no longer need to write 
                hardware-specific code. By leveraging AI to generate, validate, and optimize kernels, 
                we're building a universal abstraction layer that makes high-performance computing 
                accessible to everyone.
              </p>
              <p className="text-muted-foreground leading-relaxed">
                Our long-term vision extends beyond kernel optimization to encompass intelligent 
                system-level resource management, predictive performance modeling, and adaptive 
                runtime optimization that learns from usage patterns to continuously improve.
              </p>
            </Card>
          </section>

          {/* Creator Section */}
          <section className="space-y-6">
            <h2 className="text-3xl font-bold text-center">Creator</h2>
            <Card className="p-8 bg-card/50">
              <div className="flex flex-col md:flex-row gap-8 items-center">
                <div className="w-32 h-32 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-4xl font-bold">
                  BD
                </div>
                <div className="flex-1 text-center md:text-left space-y-4">
                  <div>
                    <h3 className="text-2xl font-bold mb-2">Bisina Daniel</h3>
                    <p className="text-muted-foreground">
                      Software Developer specializing in AI, Web Development, Cybersecurity, and System Design
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Passionate about building systems that bridge the gap between cutting-edge research 
                    and practical applications. UHop emerged from a vision to make hardware optimization 
                    accessible through AI, eliminating the need for specialized low-level programming expertise.
                  </p>
                  <div className="flex gap-3 justify-center md:justify-start">
                    <Button variant="outline" size="sm" asChild>
                      <a href="https://github.com/sevenloops" target="_blank" rel="noopener noreferrer">
                        <Github className="h-4 w-4 mr-2" />
                        GitHub
                      </a>
                    </Button>
                    <Button variant="outline" size="sm" asChild>
                      <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
                        <Linkedin className="h-4 w-4 mr-2" />
                        LinkedIn
                      </a>
                    </Button>
                    <Button variant="outline" size="sm" asChild>
                      <a href="mailto:contact@uhop.dev">
                        <Mail className="h-4 w-4 mr-2" />
                        Email
                      </a>
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          </section>

          {/* Technology Stack */}
          <section className="space-y-6">
            <h2 className="text-3xl font-bold text-center">Technology</h2>
            <Card className="p-6 bg-card/50">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-3 text-primary">Core Technologies</h3>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• Python runtime with C++ performance-critical paths</li>
                    <li>• CUDA, OpenCL, and ROCm backend support</li>
                    <li>• LLM-powered code generation pipeline</li>
                    <li>• Automated validation and benchmarking framework</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-3 text-primary">Research Areas</h3>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>• AI-driven kernel synthesis</li>
                    <li>• Cross-hardware performance modeling</li>
                    <li>• Automatic correctness verification</li>
                    <li>• Adaptive caching strategies</li>
                  </ul>
                </div>
              </div>
            </Card>
          </section>

          {/* Get Involved */}
          <section className="space-y-6">
            <h2 className="text-3xl font-bold text-center">Get Involved</h2>
            <Card className="p-8 bg-gradient-to-r from-primary/10 to-secondary/10 border-primary/20 text-center space-y-4">
              <p className="text-muted-foreground">
                UHop is open source and welcomes contributions from the community. Whether you're 
                interested in adding new operations, improving kernel generation, or optimizing for 
                new hardware platforms, we'd love to collaborate.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button size="lg" asChild>
                  <a href="https://github.com/sevenloops/uhop" target="_blank" rel="noopener noreferrer">
                    <Github className="h-5 w-5 mr-2" />
                    Contribute on GitHub
                  </a>
                </Button>
                <Button size="lg" variant="outline" asChild>
                  <a href="mailto:contact@uhop.dev">
                    Contact Us
                  </a>
                </Button>
              </div>
            </Card>
          </section>

          {/* Roadmap */}
          <section className="space-y-6">
            <h2 className="text-3xl font-bold text-center">Roadmap</h2>
            <div className="space-y-4">
              {[
                { phase: "Phase 1", title: "Core Infrastructure", status: "In Progress", items: ["Hardware detection", "Basic kernel library", "CLI tools", "Caching system"] },
                { phase: "Phase 2", title: "AI Integration", status: "Planned", items: ["LLM-based generation", "Automated validation", "Performance prediction", "Adaptive optimization"] },
                { phase: "Phase 3", title: "Ecosystem", status: "Future", items: ["Framework integrations", "Cloud deployment", "Distributed optimization", "Community kernel library"] }
              ].map((phase, idx) => (
                <Card key={idx} className="p-6 bg-card/50">
                  <div className="flex items-start gap-4">
                    <div className="w-20 shrink-0">
                      <div className="text-sm font-semibold text-primary">{phase.phase}</div>
                      <div className="text-xs text-muted-foreground mt-1">{phase.status}</div>
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold mb-2">{phase.title}</h3>
                      <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-muted-foreground">
                        {phase.items.map((item, i) => (
                          <li key={i}>• {item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default About;
