---
layout: archive
title: "AI Projects"
permalink: /aiprojects/
author_profile: true
---

{% include base_path %}

<p>
Here I collect empirical applications and code tutorials for using semi-parametric methods to study large langauge models with pair-wise preference data. 
</p>

{% for project in site.aiprojects %}
  {% include archive-single.html %}
{% endfor %}
