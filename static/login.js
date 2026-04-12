const container   = document.querySelector('.container');
const LoginLink   = document.querySelector('.SignInLink');
const RegisterLink = document.querySelector('.SignUpLink');

if (RegisterLink) RegisterLink.addEventListener('click', () => container.classList.add('active'));
if (LoginLink)    LoginLink.addEventListener('click',    () => container.classList.remove('active'));
