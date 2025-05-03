// Accordion logic
function showAccordion(label) {
    var labels = ['Anchor', 'Golden', 'Silver', 'Wrong'];
    labels.forEach(function(l) {
        var content = document.getElementById('content-' + l);
        var btn = document.getElementById('btn-' + l);
        if (l === label) {
            if (content.style.display === 'block') {
                content.style.display = 'none';
                btn.classList.remove('active');
            } else {
                labels.forEach(function(other) {
                    if (other !== l) {
                        document.getElementById('content-' + other).style.display = 'none';
                        document.getElementById('btn-' + other).classList.remove('active');
                    }
                });
                content.style.display = 'block';
                btn.classList.add('active');
            }
        }
    });
    updateRadioRequired();
}

// Only require radio buttons in the open table
function updateRadioRequired() {
    var labels = ['Anchor', 'Golden', 'Silver', 'Wrong'];
    labels.forEach(function(label) {
        var content = document.getElementById('content-' + label);
        var radios = document.querySelectorAll('input[type="radio"][name="' + label + '"]');
        radios.forEach(function(radio) {
            if (content.style.display === 'block') {
                radio.setAttribute('required', 'required');
            } else {
                radio.removeAttribute('required');
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    updateRadioRequired();
});

document.querySelectorAll('form').forEach(function(form) {
    form.addEventListener('submit', function(e) {
        if (!this.checkValidity()) {
            var labels = ['Anchor', 'Golden', 'Silver', 'Wrong'];
            labels.forEach(function(l) {
                document.getElementById('content-' + l).style.display = 'block';
                document.getElementById('btn-' + l).classList.add('active');
            });
            updateRadioRequired();
        }
    });
});

// Drag-to-scroll for .table-responsive
function enableDragScroll() {
    document.querySelectorAll('.table-responsive').forEach(function(el) {
        let isDown = false, startX, scrollLeft;
        el.addEventListener('mousedown', function(e) {
            isDown = true;
            el.classList.add('active');
            startX = e.pageX - el.offsetLeft;
            scrollLeft = el.scrollLeft;
        });
        el.addEventListener('mouseleave', function() {
            isDown = false;
            el.classList.remove('active');
        });
        el.addEventListener('mouseup', function() {
            isDown = false;
            el.classList.remove('active');
        });
        el.addEventListener('mousemove', function(e) {
            if (!isDown) return;
            e.preventDefault();
            const x = e.pageX - el.offsetLeft;
            const walk = (x - startX) * 1.7;
            el.scrollLeft = scrollLeft - walk;
        });
    });
}
enableDragScroll();

// AJAX resample
function resampleTableDirect(label) {
    var subtopic = document.querySelector('input[name="subtopic"]').value;
    fetch(`/resample_table/${label}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subtopic: subtopic })
    })
    .then(response => response.text())
    .then(html => {
        document.getElementById('table-' + label).innerHTML = html;
        updateRadioRequired();
        enableDragScroll();
    });
}
