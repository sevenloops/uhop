import { ArrowRight, Cpu, Zap, Database, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/10 via-transparent to-transparent" />
        
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="max-w-4xl mx-auto text-center space-y-8 animate-fade-in">
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight">
              <span className="gradient-text">Universal Hardware</span>
              <br />
              <span className="gradient-text">Optimization Protocol</span>
            </h1>
            
            <p className="text-xl sm:text-2xl text-muted-foreground max-w-2xl mx-auto">
              AI-Generated Kernel Intelligence — Optimize once, run anywhere.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
              <Button size="lg" asChild className="group">
                <Link to="/demo" className="flex items-center gap-2">
                  Try Demo
                  <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <a href="https://github.com/sevenloops/uhop" target="_blank" rel="noopener noreferrer">
                  View on GitHub
                </a>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link to="/docs">Read Docs</Link>
              </Button>
            </div>
          </div>
        </div>

        {/* Floating elements */}
        <div className="absolute top-1/4 left-10 w-20 h-20 rounded-full bg-primary/20 blur-3xl animate-float" />
        <div className="absolute bottom-1/4 right-10 w-32 h-32 rounded-full bg-secondary/20 blur-3xl animate-float" style={{ animationDelay: '2s' }} />
      </section>

      {/* Flow Diagram Section */}
      <section className="py-20 relative">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl sm:text-4xl font-bold text-center mb-16">
            How UHop Works
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-6xl mx-auto">
            {[
              {
                icon: Cpu,
                title: "Detect",
                description: "Automatically identifies CPU, GPU, or accelerator hardware (CUDA, OpenCL, ROCm)"
              },
              {
                icon: Zap,
                title: "Generate",
                description: "AI creates optimized kernels for fundamental operations (MatMul, Conv2D, ReLU)"
              },
              {
                icon: Target,
                title: "Optimize",
                description: "Benchmarks multiple implementations and validates correctness"
              },
              {
                icon: Database,
                title: "Cache",
                description: "Stores fastest implementation per device configuration for instant reuse"
              }
            ].map((step, index) => (
              <Card key={index} className="relative p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all group">
                <div className="absolute -top-4 left-6 w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-sm font-bold text-primary border border-primary/50">
                  {index + 1}
                </div>
                <step.icon className="h-12 w-12 text-primary mb-4 group-hover:scale-110 transition-transform" />
                <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                <p className="text-muted-foreground">{step.description}</p>
              </Card>
            ))}
          </div>

          <div className="mt-12 text-center">
            <p className="text-2xl font-semibold gradient-text">
              Detect. Generate. Optimize. Deploy.
            </p>
          </div>
        </div>
      </section>

      {/* Key Metrics Section */}
      <section className="py-20 bg-card/30">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto text-center">
            {[
              { value: "Universal", label: "From CPU to CUDA to ROCm — one protocol" },
              { value: "AI-Driven", label: "Generates and validates optimized kernels" },
              { value: "Zero Config", label: "Automatic hardware detection and optimization" }
            ].map((metric, index) => (
              <div key={index} className="space-y-2">
                <div className="text-4xl font-bold gradient-text">{metric.value}</div>
                <p className="text-muted-foreground">{metric.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl sm:text-4xl font-bold text-center mb-16">
            Core Capabilities
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {[
              {
                title: "Hardware Detection",
                description: "Automatically discovers available compute devices and their capabilities"
              },
              {
                title: "Kernel Generation",
                description: "AI-powered creation of optimized implementations for core operations"
              },
              {
                title: "Performance Benchmarking",
                description: "Real-world testing to identify the fastest kernel for your hardware"
              },
              {
                title: "Correctness Validation",
                description: "Ensures generated kernels produce accurate results before deployment"
              },
              {
                title: "Smart Caching",
                description: "Persistent storage of optimal implementations per device configuration"
              },
              {
                title: "Backend Abstraction",
                description: "Seamless support for CUDA, OpenCL, ROCm, and CPU backends"
              }
            ].map((feature, index) => (
              <Card key={index} className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/30 transition-all">
                <h3 className="text-lg font-semibold mb-3">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-primary/10 to-secondary/10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl sm:text-4xl font-bold mb-6">
            Ready to Optimize Your Hardware?
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Start using UHop to automatically detect, generate, and optimize kernels for your hardware.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" asChild>
              <Link to="/docs">Get Started</Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <a href="https://github.com/sevenloops/uhop" target="_blank" rel="noopener noreferrer">
                Star on GitHub
              </a>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
