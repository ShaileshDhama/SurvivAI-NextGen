// Generated by ReScript, PLEASE EDIT WITH CARE

import * as React from "react";

function defaultContext_toggleDarkMode() {
  
}

function defaultContext_toggleSidebar() {
  
}

var defaultContext = {
  darkMode: false,
  toggleDarkMode: defaultContext_toggleDarkMode,
  sidebarOpen: true,
  toggleSidebar: defaultContext_toggleSidebar
};

var context = React.createContext(defaultContext);

function LayoutContext$Provider(props) {
  React.useState(false);
  React.useState(true);
  return (React.createElement(
        context.Provider,
        { value: contextValue },
        children
      ));
}

var Provider = {
  make: LayoutContext$Provider
};

function useLayout() {
  return React.useContext(context);
}

export {
  defaultContext ,
  context ,
  Provider ,
  useLayout ,
}
/* context Not a pure module */
