import re

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

marketing_html = """
    <!-- Marketing Section -->
    <div class="marketing-section">
      
      <!-- Video Section -->
      <div class="hero-video-container">
        <video autoplay loop muted playsinline>
          <source src="../googlellm.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>

      <!-- Feature Section -->
      <div class="feature-grid">
        <div class="feature-text">
          <h2>Deep Understanding, Instantly</h2>
          <p>Transform any topic into a rich, interactive learning experience. Similar to NotebookLM, LEARNAI synthesizes vast amounts of information and presents it in a clear, digestible format tailored to your learning style.</p>
        </div>
        <div class="feature-img-container">
          <img src="../googlellmimage.png" alt="LEARNAI Feature Interface">
        </div>
      </div>

      <!-- Reviews Section -->
      <div class="reviews-section">
        <h2>Loved by Learners</h2>
        <div class="reviews-grid">
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"LEARNAI completely changed how I study for my engineering exams. The deep dive feature is absolutely phenomenal!"</div>
            <div class="review-author">
              <div class="review-avatar">R</div>
              <div class="review-name">Rahul Sharma</div>
            </div>
          </div>
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"The interface feels incredibly professional and smooth. It's like having a personal tutor available 24/7."</div>
            <div class="review-author">
              <div class="review-avatar">P</div>
              <div class="review-name">Priya Patel</div>
            </div>
          </div>
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"I love how it breaks down complex medical concepts into simple, visual components. Highly recommended!"</div>
            <div class="review-author">
              <div class="review-avatar">A</div>
              <div class="review-name">Amit Kumar</div>
            </div>
          </div>
          <div class="review-card">
            <div class="review-stars">★★★★★</div>
            <div class="review-text">"The speed and quality of responses are unmatched. Better than ChatGPT for focused learning."</div>
            <div class="review-author">
              <div class="review-avatar">S</div>
              <div class="review-name">Sneha Gupta</div>
            </div>
          </div>
        </div>
      </div>

    </div>
"""

if 'class="marketing-section"' not in content:
    parts = content.split('<div id="screen-loading"')
    if len(parts) == 2:
        p1 = parts[0]
        last_div_idx = p1.rfind("</div>")
        if last_div_idx != -1:
            new_p1 = p1[:last_div_idx] + marketing_html + p1[last_div_idx:]
            content = new_p1 + '<div id="screen-loading"' + parts[1]

            with open('templates/index.html', 'w', encoding='utf-8') as f:
                f.write(content)
            print("Successfully injected marketing_html!")
        else:
            print("Failed to find closing div in p1")
    else:
        print("Failed to split at screen-loading")
else:
    print("marketing_html already injected")
