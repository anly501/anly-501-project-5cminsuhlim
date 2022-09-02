/* references: 
 * https://stackoverflow.com/questions/69401316/when-i-scroll-down-the-window-it-is-not-removing-or-adding-the-active-class
 * https://stackoverflow.com/questions/23706003/changing-nav-bar-color-after-scrolling
 * https://stackoverflow.com/questions/70688456/change-navbar-colour-on-scroll-in-bootstrap-5
 * https://developer.mozilla.org/en-US/docs/Web/API/Window/scrollY
 */

const header = document.querySelector('.navbar');

window.onscroll = function() {
    if(window.scrollY > 250) {
        header.classList.add('navbar');
    }
    else {
        header.classList.remove('navbar');
    }
}
